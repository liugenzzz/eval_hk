"""Token statistics of an export_train_data.py output (field_full), with the Qwen3-VL tokenizer.

text      content of every turn, <image>/<video> placeholders removed; pt rows: the text
template  Qwen chat template, 5 tokens per turn (<|im_start|> role \\n ... <|im_end|> \\n); pt rows: +1 EOS
image     Qwen3-VL: one token per 32x32 pixels after smart_resize into the 32B training budget (image_min_pixels
          32768, image_max_pixels 262144, i.e. 32..256 tokens), +2 for vision start/end; mean over up to 2,000
          sampled images of the file x the file's image count
video     frames = min(32, duration x 1 fps), even (video_fps 1, video_maxlen 32, temporal patch 2); every 2-frame
          group gives h x w / 1024 tokens after smart_resize into video_max_pixels 16384, +2; duration and size read
          with PyAV from up to 600 sampled clips per file (per-group timestamp text is not counted)
Files up to EXACT_BYTES are read in full (exact counts). Larger ones (the books_combined CPT files) are sampled:
SAMPLE rows picked by byte position (probability proportional to the row's length); total = bytes x mean(tokens/bytes)
(Hansen-Hurwitz), reported with a 95% interval.

  python token_stats.py <export dir>        # writes <dir>/_logs/token_stats.json, prints summary tables
"""
import collections
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from tokenizers import Tokenizer

TOKENIZER = "/mnt/si003010kcx0/mmdata/mmodel/Qwen3_VL_32BInstruct/tokenizer.json"
EXACT_BYTES = 2 * 10 ** 9
SAMPLE = 4000
IMG_SAMPLE, VID_SAMPLE = 2000, 600
IMG_MIN, IMG_MAX, VID_MAX, FACTOR = 32768, 262144, 16384, 32
PLACEHOLDER = re.compile(r"<image>|<video>")
TOK = Tokenizer.from_file(TOKENIZER)


def ntokens(texts):
    return [len(e.ids) for e in TOK.encode_batch(texts, add_special_tokens=False)] if texts else []


def smart_resize(h, w, min_pixels, max_pixels, factor=FACTOR):
    h_bar, w_bar = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar, w_bar = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def image_tokens(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        return None
    hb, wb = smart_resize(h, w, IMG_MIN, IMG_MAX)
    return hb * wb // (FACTOR * FACTOR) + 2


def video_tokens(path):
    import av
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            dur = float(c.duration) / 1e6 if c.duration else float(s.duration * s.time_base)
            w, h = s.width, s.height
    except Exception:
        return None
    frames = max(2, min(32, int(dur * 1.0)))
    frames += frames % 2
    hb, wb = smart_resize(h, w, FACTOR * FACTOR, VID_MAX)
    return frames // 2 * (hb * wb // (FACTOR * FACTOR)) + 2


def row_parts(rec):
    """-> (texts to tokenize, turns, images, videos)"""
    if "text" in rec:
        return [rec["text"]], 0, [], []
    texts = [PLACEHOLDER.sub("", m["value"]) for m in rec["conversations"]]
    return texts, len(texts), rec.get("images") or [], rec.get("videos") or []


def line_at(f, off):
    """The whole line containing byte offset off (f opened in binary mode)."""
    back = 1 << 16
    while True:
        start = max(0, off - back)
        f.seek(start)
        buf = f.read(off - start)
        i = buf.rfind(b"\n")
        if i >= 0 or start == 0:
            f.seek(start + i + 1 if i >= 0 else 0)
            return f.readline()
        back *= 4


def stats_file(path, size, rng):
    st = collections.Counter()
    imgs, vids = [], []
    method = "exact" if size <= EXACT_BYTES else "sampled"
    if method == "exact":
        batch, meta = [], []

        def flush():
            lens = ntokens(batch)
            k = 0
            for nt, turns in meta:
                st["text_tokens"] += sum(lens[k:k + nt])
                k += nt
            batch.clear()
            meta.clear()
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts, turns, im, vi = row_parts(rec)
                st["rows"] += 1
                st["turns"] += turns
                st["content_bytes"] += sum(len(t.encode("utf-8")) for t in texts)
                st["images"] += len(im)
                st["videos"] += len(vi)
                for p in im:      # reservoir sample of image / video paths
                    if len(imgs) < IMG_SAMPLE:
                        imgs.append(p)
                    elif rng.random() < IMG_SAMPLE / st["images"]:
                        imgs[rng.randrange(IMG_SAMPLE)] = p
                for p in vi:
                    if len(vids) < VID_SAMPLE:
                        vids.append(p)
                    elif rng.random() < VID_SAMPLE / st["videos"]:
                        vids[rng.randrange(VID_SAMPLE)] = p
                batch.extend(texts)
                meta.append((len(texts), turns))
                if len(batch) >= 4096:
                    flush()
            flush()
        st["template_tokens"] = 5 * st["turns"] + (st["rows"] if st["turns"] == 0 else 0)
        out = dict(st)
    else:
        ratios, inv, turns_r, cbytes_r = [], [], [], []
        with open(path, "rb") as f:
            lines = [line_at(f, rng.randrange(size)) for _ in range(SAMPLE)]
        recs = [json.loads(x) for x in lines]
        parts = [row_parts(r) for r in recs]
        for (texts, turns, im, vi), raw in zip(parts, lines):
            b = len(raw)
            t = sum(ntokens(texts))
            ratios.append(t / b)
            inv.append(1 / b)
            turns_r.append(turns / b)
            cbytes_r.append(sum(len(x.encode("utf-8")) for x in texts) / b)
        n = len(ratios)
        mean = sum(ratios) / n
        sd = math.sqrt(sum((r - mean) ** 2 for r in ratios) / (n - 1))
        rows_est = size * sum(inv) / n
        out = {"rows_est": round(rows_est), "text_tokens": round(size * mean),
               "text_tokens_ci95": [round(size * (mean - 1.96 * sd / math.sqrt(n))), round(size * (mean + 1.96 * sd / math.sqrt(n)))],
               "content_bytes": round(size * sum(cbytes_r) / n), "turns": round(size * sum(turns_r) / n),
               "images": 0, "videos": 0, "sample_rows": n}
        out["template_tokens"] = 5 * out["turns"] + (out["rows_est"] if out["turns"] == 0 else 0)
    if imgs:
        with ThreadPoolExecutor(32) as ex:
            got = [x for x in ex.map(image_tokens, imgs) if x is not None]
        out["image_tokens_per_image"] = round(sum(got) / len(got), 1) if got else None
        out["image_tokens"] = round(out["images"] * sum(got) / len(got)) if got else None
    if vids:
        with ThreadPoolExecutor(16) as ex:
            got = [x for x in ex.map(video_tokens, vids) if x is not None]
        out["video_tokens_per_video"] = round(sum(got) / len(got), 1) if got else None
        out["video_tokens"] = round(out["videos"] * sum(got) / len(got)) if got else None
    out["method"] = method
    return out


def main():
    root = sys.argv[1]
    man = json.load(open(os.path.join(root, "manifest.json"), encoding="utf-8"))
    rng = random.Random(20261010)
    t0 = time.time()
    res = {}
    for name, e in sorted(man["files"].items(), key=lambda x: x[1]["bytes"]):
        s = stats_file(os.path.join(root, name), e["bytes"], rng)
        s.update(bytes=e["bytes"], manifest_rows=e["rows"], corpus_type=e["corpus_type"], domain=e.get("domain", ""),
                 kind=e["kind"], task_type=e.get("task_type"))
        res[name] = s
        print(f"{time.time() - t0:7.0f}s {s['method']:7s} {name:70s} text {s['text_tokens']:>14,}"
              f"  img {s.get('image_tokens') or 0:>12,}  vid {s.get('video_tokens') or 0:>12,}", flush=True)
    out = os.path.join(root, "_logs", "token_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "tokenizer": TOKENIZER, "files": res}, f,
                  ensure_ascii=False, indent=1)
    print("written", out, f"{(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()

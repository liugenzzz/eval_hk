"""离线跑通链路用的**假** VLM 服务 —— 产出的分数没有任何评估意义。

数据构建端（liugenzzz/target_detection_vl_dataset）要调一个 VLM 来挑目标、写描述。
没有那个服务的机器上，用这个假服务顶上，就能把整条链路跑一遍：

    # 1. 拉公开数据（构建端的脚本，548 图 / 38759 框）
    python scripts/get_visdrone.py --out ./data/visdrone

    # 2. 起假服务
    python scripts/fake_vlm_server.py 18899 &

    # 3. 构建端 config/local.yaml 指过去，然后 python scripts/build.py
    #      vlm: {enabled: true, api_url: "http://127.0.0.1:18899/v1/chat/completions",
    #            model: "fake-vlm", stream: false}

    # 4. 拿产出的 test.jsonl 跑本项目的评估器

它按 vlm_select.txt 的契约作答：从提示词里解析出框列表和描述类型指派，返回格式合法的
JSON。**描述文本是套模板的**，所以：

- 可以用来验证代码通不通、报表出不出、改动有没有把链路弄坏
- **不能**用它跑出来的分数说明任何模型质量问题

这个脚本抓到过三个真 bug（连字符类别名被截断、短答案的模板包装、报表里的空格子），
都是单测覆盖不到、只有真跑一遍数据才会暴露的。
"""

import json, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BOX = re.compile(r"^\s*\[(\d+)\]\s+(\S+)\s+位于", re.MULTILINE)
SLOT = re.compile(r"第 (\d+) 个挑中的目标 -> 「([a-z_]+)」")

# 每种 kind 的答案都躲开它自己的 must-not 词表，且 >= 18 字（min_description_len）
DESC = {
    "appearance": ("这{mw}{label}本身长什么样？",
                   "整体呈深灰色调，轮廓清晰完整，车身线条平直，表面有明显的反光质感，细节可辨。"),
    "state":      ("这{mw}{label}当前处于什么状态？",
                   "当前保持静止未移动，朝向与道路方向一致，没有正在行进或作业的迹象，状态稳定。"),
    "part":       ("这{mw}{label}最显眼的那个部位是什么样的？",
                   "最显眼的是它上半部的结构，边缘平直，与主体连接紧密，表面颜色比主体略深一些。"),
    "position":   ("这{mw}{label}在画面的什么方位？",
                   "在画面偏中间的位置，大致处在从左往右三分之一、从上往下二分之一的地方。"),
    "relation":   ("这{mw}{label}和周围的物体是什么关系？",
                   "紧挨着同一排的另一个目标，两者之间几乎没有空隙，前方不远处还有一个同类。"),
    "contrast":   ("图中还有别的同类，这{mw}{label}和它们有什么不同？",
                   "这一个的轮廓比另外几个更完整，占据的范围也更大，朝向和其余几个明显不同。"),
    "full":       ("描述这{mw}{label}的外观、所在方位，以及它旁边有什么。",
                   "一个深灰色调、轮廓完整的目标，处在画面中部偏左，紧挨着同一排的另一个同类。"),
}
ATTR = ["深灰色", "轮廓完整", "朝向左侧", "位于中部", "体积较大", "颜色偏深"]


def reply(prompt: str) -> str:
    boxes = BOX.findall(prompt)
    slots = [k for _, k in sorted(SLOT.findall(prompt), key=lambda x: int(x[0]))]
    if not boxes:
        return json.dumps({"picked": []}, ensure_ascii=False)
    picked = []
    for i, (idx, label) in enumerate(boxes[:max(1, len(slots))]):
        kind = slots[i] if i < len(slots) else "full"
        q, a = DESC.get(kind, DESC["full"])
        attribute = ATTR[i % len(ATTR)]
        picked.append({
            "id": int(idx),
            "attribute": attribute,
            "color": "深灰色",
            "questions": [f"框出图中{attribute}的{label}。",
                          f"输出{attribute}的{label}的检测框。",
                          f"请给出{attribute}的{label}的边界框。"],
            "describe_q": q.format(mw="个", label=label),
            "description": a,
        })
    return json.dumps({"picked": picked}, ensure_ascii=False)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        text = ""
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                text += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
        out = {"choices": [{"message": {"role": "assistant", "content": reply(text)},
                            "finish_reason": "stop"}]}
        payload = json.dumps(out, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()

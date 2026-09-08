import base64
import io
import logging
import math
import os
import os.path as osp
import re
import ast
from urllib.request import urlopen

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from vlmeval.dataset.utils import build_judge, levenshtein_distance
from vlmeval.smp import (decode_base64_to_image_file, dump, encode_image_to_base64,
                         get_intermediate_file_path, get_logger, listinstr, load, read_ok,
                         toliststr)
from .image_base import ImageBaseDataset

logger = get_logger(__name__)

FAIL_MSG = 'Failed to obtain answer via API.'


def get_gpt4_ICE():
    example_1 = """
---
Question: List the primary questions asked about the services in this report.
Analysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\n\n
1. Is the service safe?\n
2. Is the service effective?\n
3. Is the service caring?\n
4. Is the service responsive?\n
5. Is the service well-led?
Extracted answer: [
    'Is the servife safe?',
    'Is the service effective',
    'Is the serve caring?',
    'Is the service responsive?',
    'Is the service well-led?'
]
Answer format: List\n
"""

    example_2 = """
---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities)
Regulations in total. Here are the specifics:\n\n1. Regulation 13: Safeguarding service users from abuse and
improper treatment\n2. Regulation 12: Safe care and treatment\n3. Regulation 18: Staffing\n4. Regulation 11:
Need for consent\n5. Regulation 10: Dignity and respect\n6. Regulation 9: Person-centred care\n7. Regulation 17:
Good governance\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\n9.
Regulation 18: Failure to maintain an accurate and up-to-date care plan\n10. Regulation 11: Failure to implement
the Mental Capacity Act 2005 code of practice effectively\n\nThese breaches involve issues concerning staffing,
safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to
notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer\n
"""

    example_3 = """
---
Question: According to the survey that is the percentage of Chinese who are paying more or
about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying
more or about the same attention to politics after Trump's election. The report focuses primarily on American
demographics and does not include specific details about the Chinese population in relation to this question. If
you need information about a different demographic or a summary of the findings from the American demographic,
I can certainly help with that!
Extracted answer: Not answerable
Answer format: String\n
"""

    example_4 = """
---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the
text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be
able to help you with your question.
Extracted answer: Fail to answer
Answer format: String\n
"""

    return [example_1, example_2, example_3, example_4]

def build_mmlongbench_gpt4_prompt(line):
    answer_format = str(line.get('answer_format', '')).strip()
    task_description = """
Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List.
- Use the target answer format if it is provided. Do not change the answer into another format.
- Extract the final answer only. If the analysis explores several candidates and then concludes with a final answer,
use the final concluded answer instead of intermediate candidates.
- Do not judge whether the answer is correct. Only extract the answer stated by the analysis.
If you find the analysis the question can not be answered from the given documents, type "Not answerable".
Exception: If the analysis only tells you that it can not read/understand the images or documents,
type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```
Please read the following example, then extract the answer from the model response
and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example
    prompt += '---\nQuestion:' + question + '\n'
    if answer_format:
        prompt += 'Target answer format: ' + answer_format + '\n'
    prompt += 'Analysis: ' + prediction
    return prompt

def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip('%').strip())
    try:
        prediction = float(str(prediction).strip().rstrip('%').strip())
    except Exception:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if math.isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def _safe_literal_eval(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _unwrap_singleton_list(value):
    parsed = _safe_literal_eval(value)
    if isinstance(parsed, list) and len(parsed) == 1:
        return parsed[0]
    return value


def _strip_list_prefix(s):
    # MMLongBench list answers often include section labels in model outputs
    # (e.g. "A. Iterative Retrieval") while references omit them.
    return re.sub(r'^\s*(?:[\(\[]?[a-zA-Z][\)\].:-]|[\(\[]?\d+[\)\].-])\s+', '', str(s)).strip()


def get_clean_string(s, strip_list_prefix=False):
    s = _unwrap_singleton_list(s)
    if strip_list_prefix:
        s = _strip_list_prefix(s)
    s = str(s).lower().strip()
    if s.endswith('miles'):
        s = s[:-5].strip()
    elif s.endswith('mile'):
        s = s[:-4].strip()
    if s.endswith('million'):
        s = s[:-7].strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", '', s).strip()
    s = s.strip().lstrip('$').strip()
    s = s.strip().rstrip('%').strip()
    s = s.replace('modelling', 'modeling')
    s = re.sub(r'\s*-\s*', '-', s)
    s = re.sub(r'\s*:\s*', ':', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.rstrip('.,;:!?').strip()
    return s

def is_exact_match(s):
    flag = False
    # Website
    if 'https://' in s:
        flag = True
    # code file
    if s.endswith('.py') or s.endswith('ipynb'):
        flag = True
    if s.startswith('page'):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if 'a.m.' in s or 'p.m.' in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


_NUM_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
    'ninety': 90, 'hundred': 100,
}


def _parse_int(value):
    value = get_clean_string(value)
    try:
        return int(float(value.replace(',', '')))
    except Exception:
        pass
    tokens = re.split(r'[\s-]+', value)
    if not tokens or any(tok not in _NUM_WORDS for tok in tokens):
        raise ValueError(f'Cannot parse int: {value}')
    current = 0
    for tok in tokens:
        num = _NUM_WORDS[tok]
        if num == 100:
            current = max(current, 1) * 100
        else:
            current += num
    return current


def _parse_float(value):
    value = get_clean_string(value).replace(',', '')
    match = re.search(r'[-+]?\d+(?:\.\d+)?', value)
    if match:
        return float(match.group(0))
    return float(value)


def _parse_list(value):
    parsed = _safe_literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(value, str):
        text = value.strip()
        if '\n' in text:
            items = [re.sub(r'^\s*[-*]\s+', '', x).strip() for x in text.splitlines()]
            items = [x for x in items if x]
            if len(items) > 1:
                return items
        if ';' in text:
            return [x.strip() for x in text.split(';') if x.strip()]
    return [value]

def _best_list_score(gt, pred):
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0

    gt = [get_clean_string(a, strip_list_prefix=True) for a in gt]
    pred = [get_clean_string(a, strip_list_prefix=True) for a in pred]
    if isfloat(gt[0]) or is_exact_match(gt[0]):
        gt_norm = sorted(gt)
        pred_norm = sorted(pred)
        if len(gt_norm) == len(pred_norm):
            return float('-'.join(gt_norm) == '-'.join(pred_norm))
        common = sum(1 for x in pred_norm if x in gt_norm)
        return common / max(len(gt_norm), len(pred_norm))

    # Greedy maximum matching is enough for the short lists in MMLongBench and
    # avoids penalizing equivalent items because they appear in a different order.
    unmatched = pred[:]
    scores = []
    for gt_item in gt:
        if not unmatched:
            scores.append(0.0)
            continue
        cand_scores = [anls_compute(gt_item, pred_item) for pred_item in unmatched]
        best_idx = max(range(len(cand_scores)), key=lambda i: cand_scores[i])
        scores.append(cand_scores[best_idx])
        unmatched.pop(best_idx)
    if len(pred) != len(gt):
        return sum(scores) / max(len(gt), len(pred))
    return min(scores)


def get_font():
    try:
        truetype_url = "https://opencompass.openxlab.space/utils/Fonts/SimHei.ttf"
        ff = urlopen(truetype_url)
        font = ImageFont.truetype(ff, size=40)
    except Exception as e:
        logging.warning(f'{type(e)}: {e}')
        logging.warning("Fail to download the font. Use the default one.")
        font = ImageFont.load_default(size=40)
    return font


def frame2img(img_path_list, font, save_path=None, idx_start=0, target_edge=1120):
    imgs = [Image.open(img_path) for img_path in img_path_list]

    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w / h
        if w > h:
            new_w = target_edge
            new_h = int(target_edge / scale)
        else:
            new_w = int(target_edge * scale)
            new_h = target_edge
        img = transforms.functional.resize(img, [new_h, new_w],)
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w, h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        new_img = Image.new("RGB", (new_w, new_h), "white")
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h), f"<IMAGE {idx + idx_start}>", font=font, fill="black")
            if idx + 1 < len(imgs):
                draw.line([(0, pad + curr_h + h + 5), (new_w, pad + curr_h + h + 5)], fill='black', width=2)
            curr_h += h + 10 + pad
    else:
        for im in imgs:
            w, h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f"<IMAGE {idx + idx_start}>", font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill='black', width=2)
            curr_w += w + 10

    if save_path is not None:
        new_img.save(save_path)

    return new_img


def concat_images(image_list, max_concat=1, column_num=1, target_edge=1120, max_column_num=20):
    concatenated_images = []
    if column_num == -1:
        MAX_COLUMN_NUM = max_column_num
        max_concat = 1
        while len(image_list) / max_concat > MAX_COLUMN_NUM:
            max_concat += 1
        interval = max(math.ceil(len(image_list) / max_concat), 1)
        for i in range(0, len(image_list), interval):
            batch_images = image_list[i:i + interval]
            concatenated_image = frame2img(batch_images, font=get_font(), idx_start=i, target_edge=target_edge)
            concatenated_images.append(concatenated_image)
    else:
        interval = max(math.ceil(len(image_list) / max_concat), 1)
        for i in range(0, len(image_list), interval):
            batch_images = [Image.open(filename) for filename in image_list[i:i + interval]]
            if column_num == 1:
                total_height = batch_images[0].height * len(batch_images)
            else:
                total_height = batch_images[0].height * ((len(batch_images) - 1) // column_num + 1)
            concatenated_image = Image.new('RGB', (batch_images[0].width * column_num, total_height), 'white')

            x_offset, y_offset = 0, 0
            for count, image in enumerate(batch_images):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                if (count + 1) % column_num == 0:
                    y_offset += image.height
                    x_offset = 0
            concatenated_images.append(concatenated_image)
    return concatenated_images


def eval_score(gt, pred, answer_type):
    answer_type = str(answer_type).strip()
    if answer_type == 'Int':
        try:
            gt, pred = _parse_int(gt), _parse_int(pred)
        except Exception:
            pred = ''
        score = (gt == pred)
    elif answer_type == 'Float':
        try:
            gt = _parse_float(gt)
            pred = _parse_float(pred)
        except Exception:
            pred = ''
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type == 'Str':
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)
    else:
        gt = _parse_list(gt)
        pred = _parse_list(pred)
        score = _best_list_score(gt, pred)

    return float(score)


def _parse_extracted_answer(res):
    text = str(res or '').strip()
    text = re.sub(r'^```(?:\w+)?\s*|\s*```$', '', text, flags=re.S).strip()
    patterns = [
        r'Extracted answer\s*[:：]\s*(.*?)(?:\n\s*Answer format\s*[:：]|$)',
        r'Answer\s*[:：]\s*(.*?)(?:\n\s*Answer format\s*[:：]|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return match.group(1).strip()
    return text.strip()

def MMLongBench_auxeval(model, line):
    prompt = build_mmlongbench_gpt4_prompt(line)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            pred = _parse_extracted_answer(res)
            return dict(log=log, res=res, pred=pred)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='', pred='')


def get_f1(data):
    gt_pos_data = data[data.apply(lambda k: k['answer'] != 'Not answerable', axis=1)]
    pred_pos_data = data[data.apply(lambda k: k['pred'] != 'Not answerable', axis=1)]
    recall = sum(gt_pos_data['score'].tolist()) / len(gt_pos_data)
    precision = sum(pred_pos_data['score'].tolist()) / len(pred_pos_data) if len(pred_pos_data) else 0.0
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def MMLongBench_acc(result_file):
    data = load(result_file)
    overall_score = 0.0
    score_list = list()
    for i in range(len(data)):
        item = data.iloc[i]
        try:
            score = eval_score(item['answer'], item['pred'], item['answer_format'])
        except Exception:
            score = 0.0
        score_list.append(score)
        overall_score += score

    data['score'] = score_list
    dump(data, result_file)

    data_chart = data[data.apply(lambda k: 'Chart' in eval(k['evidence_sources']), axis=1)]
    data_table = data[data.apply(lambda k: 'Table' in eval(k['evidence_sources']), axis=1)]
    data_image = data[data.apply(lambda k: 'Figure' in eval(k['evidence_sources']), axis=1)]
    data_text = data[data.apply(lambda k: 'Pure-text (Plain-text)' in eval(k['evidence_sources']), axis=1)]
    data_layout = data[data.apply(lambda k: 'Generalized-text (Layout)' in eval(k['evidence_sources']), axis=1)]

    data_single = data[data.apply(lambda k: len(eval(k['evidence_pages'])) == 1, axis=1)]
    data_multi = data[data.apply(lambda k: len(eval(k['evidence_pages'])) > 1, axis=1)]
    data_unans = data[data.apply(lambda k: len(eval(k['evidence_pages'])) == 0, axis=1)]

    res = dict()
    res['category'] = [
        'overall_f1', 'overall_acc', 'text', 'layout', 'table', 'chart',
        'image', 'single-page', 'multi-page', 'unanswerable'
    ]
    res['num'] = [
        len(data), len(data), len(data_text), len(data_layout), len(data_table),
        len(data_chart), len(data_image), len(data_single), len(data_multi), len(data_unans)
    ]
    res['avg_score'] = [
        get_f1(data),
        overall_score / len(data),
        sum(data_text['score'].tolist()) / len(data_text) if len(data_text) > 0 else 0.0,
        sum(data_layout['score'].tolist()) / len(data_layout) if len(data_layout) > 0 else 0.0,
        sum(data_table['score'].tolist()) / len(data_table) if len(data_table) > 0 else 0.0,
        sum(data_chart['score'].tolist()) / len(data_chart) if len(data_chart) > 0 else 0.0,
        sum(data_image['score'].tolist()) / len(data_image) if len(data_image) > 0 else 0.0,
        sum(data_single['score'].tolist()) / len(data_single) if len(data_single) > 0 else 0.0,
        sum(data_multi['score'].tolist()) / len(data_multi) if len(data_multi) > 0 else 0.0,
        sum(data_unans['score'].tolist()) / len(data_unans) if len(data_unans) > 0 else 0.0,
    ]
    res = pd.DataFrame(res)
    return res


class MMLongBench(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'MMLongBench_DOC': 'https://opencompass.openxlab.space/utils/VLMEval/MMLongBench_DOC.tsv',
    }
    DATASET_MD5 = {
        'MMLongBench_DOC': '75f5d29965d0db68254993f6170da7c2',
    }

    SUPPORTED_MODELS = {
        'GPT4': (1, 1),
        'GPT4V': (1, 1),
        'GPT4V_HIGH': (1, 1),
        'GPT4o': (1, 1),
        'GPT4o_HIGH': (1, 1),
        'GPT4o_MINI': (1, 1),
        'MiniCPM-Llama3-V-2_5': (1, 5),
        'InternVL-Chat-V1-5': (5, 2),
        'XComposer2_4KHD': (1, 5),
        'XComposer2d5': (1, -1),
        # Enable local Qwen3-VL evaluation on MMLongBench_DOC.
        # Use smaller page grids by default; very wide all-page strips hurt OCR fidelity.
        'qwen3_vl_8b_instruct': (5, 2),
    }

    def __init__(self, dataset, **kwargs):
        self.model_list = list(self.SUPPORTED_MODELS.keys())
        model_name = kwargs['model']
        if not listinstr(self.model_list, model_name):
            raise AssertionError("{} doesn't support the evaluation on MMLongBench_DOC.".format(model_name))
        super(MMLongBench, self).__init__(dataset)

        self.is_api = True if listinstr(['GPT4'], model_name) else False
        self.max_pages = int(os.environ.get('MMLONGBENCH_MAX_PAGES', 120))
        self.pdf_dpi = int(os.environ.get('MMLONGBENCH_PDF_DPI', 144))
        self.concat_target_edge = int(os.environ.get('MMLONGBENCH_CONCAT_EDGE', 1120))
        self.max_column_num = int(os.environ.get('MMLONGBENCH_MAX_COLUMN_NUM', 20))
        concat_num, column_num = self.SUPPORTED_MODELS.get(model_name)
        self.concat_num = int(os.environ.get('MMLONGBENCH_CONCAT_NUM', concat_num))
        self.column_num = int(os.environ.get('MMLONGBENCH_COLUMN_NUM', column_num))

    def dump_image(self, origin_line):
        os.makedirs(self.img_root, exist_ok=True)

        line = origin_line.copy()
        line['image_path'] = line['image_path'][:self.max_pages]
        skip_pdf_parse = True
        for im_name in line['image_path']:
            path = osp.join(self.img_root, im_name)
            if not read_ok(path):
                skip_pdf_parse = False
                break

        # Just for being compatible with the zooped loop: zip(line['image'], line['image_path'])
        if skip_pdf_parse:
            line['image'] = line['image_path']
        else:
            try:
                import fitz
            except Exception as e:
                logging.critical(f'{type(e)}: {e}')
                raise ModuleNotFoundError(
                    'PyMuPDF is required for MMLongBench_DOC PDF parsing. '
                    'Please install it with: pip install pymupdf'
                ) from e

            pdf_data = base64.b64decode(line['image'])
            pdf_file = io.BytesIO(pdf_data)
            encoded_images = []
            with fitz.open(stream=pdf_file, filetype='pdf') as doc:
                doc = doc[:self.max_pages]
                for page in doc:
                    image = page.get_pixmap(dpi=self.pdf_dpi)
                    image_file = io.BytesIO(image.tobytes(output='png'))
                    image = Image.open(image_file)
                    encoded_image = encode_image_to_base64(image)
                    encoded_images.append(encoded_image)
            line['image'] = encoded_images
            print('process {}'.format(line['doc_id']))

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        if self.concat_num > 0 and not self.is_api:
            concatenated_images = concat_images(
                tgt_path,
                max_concat=self.concat_num,
                column_num=self.column_num,
                target_edge=self.concat_target_edge,
                max_column_num=self.max_column_num,
            )

            old_tgt_path = tgt_path
            assert isinstance(old_tgt_path, list)
            if self.column_num != -1:
                tgt_path = [
                    '_'.join(old_tgt_path[0].split('_')[:-1]) + '_concat{}_{}.jpg'.format(self.concat_num, i)
                    for i in range(len(concatenated_images))
                ]
            else:
                tgt_path = [
                    '_'.join(old_tgt_path[0].split('_')[:-1]) + '_concat_all_{}.jpg'.format(i)
                    for i in range(len(concatenated_images))
                ]

            for path, concatenated_image in zip(tgt_path, concatenated_images):
                if not read_ok(path):
                    decode_base64_to_image_file(encode_image_to_base64(concatenated_image), path)
                    num_images, image_size = len(old_tgt_path), concatenated_image.size
                    print('concat {} images to a new one with size {}. save at {}'.format(num_images, image_size, path))
        return tgt_path

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs['model']

        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')

        force_rejudge = os.environ.get('MMLONGBENCH_FORCE_REJUDGE', '0') == '1'
        if osp.exists(storage) and not force_rejudge:
            logger.warning(f'GPT scoring file {storage} already exists, will reuse it in MMLongBench_eval. ')
        else:
            if osp.exists(storage) and force_rejudge:
                logger.warning(f'MMLONGBENCH_FORCE_REJUDGE=1, will rebuild GPT scoring file {storage}.')
            data = load(eval_file)
            judge_max_tokens = int(os.environ.get('MMLONGBENCH_JUDGE_MAX_TOKENS', 512))
            model = build_judge(max_tokens=judge_max_tokens, **judge_kwargs)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file) and not force_rejudge:
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = list()
                for model, line in tqdm(tups):
                    res = MMLongBench_auxeval(model, line)
                    new_results.append(res)
                for idx, res in zip(indices, new_results):
                    ans[idx] = res
                dump(ans, tmp_file)

            log_map, res_map, pred_map = {}, {}, {}
            all_inds = [line['index'] for line in lines]
            for k in all_inds:
                v = ans[k]
                log_map[k] = v['log']
                res_map[k] = v['res']
                pred_map[k] = v['pred']
            data['res'] = [res_map[idx] for idx in data['index']]
            data['log'] = [log_map[idx] for idx in data['index']]
            data['pred'] = [pred_map[idx] for idx in data['index']]
            dump(data, storage)

        score = MMLongBench_acc(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')

        dump(score, score_pth)
        logger.info(f'MMLongBench_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
        logger.info('Score: ')
        logger.info(score)
        return score

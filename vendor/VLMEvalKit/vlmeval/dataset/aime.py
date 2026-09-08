import re
from os import path as osp

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .text_base import TextBaseDataset


def extract_aime_answer(text):
    """Extract the final AIME-style integer answer from a model response."""
    text = str(text)

    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        boxed_nums = re.findall(r'-?\d+', boxed[-1])
        if boxed_nums:
            return str(int(boxed_nums[-1]))

    answer_patterns = [
        r'(?:final answer|answer|答案)\s*(?:is|:|：)?\s*(-?\d+)',
        r'=\s*(-?\d+)\s*\.?\s*$',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(int(match.group(1)))

    nums = re.findall(r'-?\d+', text)
    return str(int(nums[-1])) if nums else ''


class AIME2025Dataset(TextBaseDataset):
    TYPE = 'QA'
    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['AIME2025']

    def load_data(self, dataset):
        return load(osp.join(LMUDataRoot(), f'{dataset}.tsv'))

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        prompt = (
            f"{line['question']}\n\n"
            "Solve the problem and give the final answer as an integer from 0 to 999. "
            "Put only the final integer in the final answer."
        )
        return [dict(type='text', value=prompt)]

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data['answer'] = [str(int(str(x).strip())) for x in data['answer']]
        data['prediction'] = [str(x) for x in data['prediction']]
        data['extracted_prediction'] = [extract_aime_answer(x) for x in data['prediction']]
        data['hit'] = data['extracted_prediction'] == data['answer']

        detail_file = get_intermediate_file_path(eval_file, '_aime2025_eval')
        dump(data, detail_file)

        ret = pd.DataFrame(
            {
                'Overall': [round(float(data['hit'].mean()) * 100, 2)],
                'Correct': [int(data['hit'].sum())],
                'Total': [int(len(data))],
            }
        )
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(ret, score_file)
        return ret

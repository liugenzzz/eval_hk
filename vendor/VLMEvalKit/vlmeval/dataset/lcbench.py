import base64
import binascii
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import warnings
import zlib
from os import path as osp

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .text_base import TextBaseDataset


def extract_python_code(response):
    response = str(response)
    fenced = re.findall(r'```(?:python|py)?\s*(.*?)```', response, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    return response.strip()


def normalize_output(text):
    return '\n'.join(str(text).replace('\r\n', '\n').replace('\r', '\n').strip().splitlines())


def _decode_base64_with_padding(encoded_text):
    s = str(encoded_text).strip()
    if not s:
        raise ValueError('empty base64 payload')

    # Remove whitespace/newlines that may be introduced by serialization.
    s = ''.join(s.split())

    candidates = [s]
    if '-' in s or '_' in s:
        candidates.append(s.replace('-', '+').replace('_', '/'))

    errors = []
    for candidate in candidates:
        padded = candidate + '=' * (-len(candidate) % 4)
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                return decoder(padded)
            except (binascii.Error, ValueError) as e:
                errors.append(e)

    raise ValueError(f'base64 decode failed after padding attempts: {errors[-1] if errors else "unknown error"}')


def decode_private_tests(encoded):
    if not encoded or pd.isna(encoded):
        return []

    raw = _decode_base64_with_padding(encoded)

    # Standard path: zlib-compressed pickle payload.
    try:
        decompressed = zlib.decompress(raw)
    except zlib.error:
        decompressed = raw

    try:
        tests = pickle.loads(decompressed)
    except Exception:
        # Fallback for plain JSON payloads.
        tests = decompressed.decode('utf-8')

    return json.loads(tests) if isinstance(tests, str) else tests


def load_tests(row, include_private=True):
    tests = []
    public_tests = row.get('public_test_cases', '[]')
    if public_tests and not pd.isna(public_tests):
        tests.extend(json.loads(public_tests))
    if include_private:
        tests.extend(decode_private_tests(row.get('private_test_cases', '')))
    return tests


def run_python_code(code, test_input, timeout):
    with tempfile.TemporaryDirectory() as tmpdir:
        src = osp.join(tmpdir, 'solution.py')
        with open(src, 'w', encoding='utf-8') as f:
            f.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, src],
                input=str(test_input),
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return False, 'timeout'

    if proc.returncode != 0:
        return False, f'runtime_error: {proc.stderr[-500:]}'
    return True, proc.stdout


class LCBenchV6Dataset(TextBaseDataset):
    TYPE = 'QA'
    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['LCBenchV6']

    def load_data(self, dataset):
        return load(osp.join(LMUDataRoot(), f'{dataset}.tsv'))

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [dict(type='text', value=line['question'])]

    def evaluate(self, eval_file, **judge_kwargs):
        timeout = int(os.environ.get('LCBENCH_TIMEOUT', judge_kwargs.pop('timeout', 5)))
        include_private = os.environ.get('LCBENCH_PUBLIC_ONLY', '0') != '1'

        # Excel cells are capped at 32767 chars and may truncate private_test_cases.
        if include_private and str(eval_file).lower().endswith('.xlsx'):
            include_private = False
            warnings.warn(
                'LCBenchV6 eval_file is .xlsx; private_test_cases may be truncated. '
                'Falling back to public tests only. To use private tests, rerun inference with PRED_FORMAT=tsv.'
            )

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]

        rows = []
        for _, row in data.iterrows():
            code = extract_python_code(row['prediction'])
            passed = 0
            error = ''

            try:
                tests = load_tests(row, include_private=include_private)
            except Exception as e:
                tests = []
                error = f'testcase_decode_error: {e}'

            for case in tests:
                ok, out_or_err = run_python_code(code, case.get('input', ''), timeout)
                expected = normalize_output(case.get('output', ''))
                actual = normalize_output(out_or_err)
                if ok and actual == expected:
                    passed += 1
                else:
                    error = out_or_err if not ok else f'wrong_answer: expected={expected!r}, actual={actual!r}'
                    break

            rows.append(
                {
                    'index': row['index'],
                    'question_id': row.get('question_id', ''),
                    'difficulty': row.get('difficulty', ''),
                    'num_tests': len(tests),
                    'passed_tests': passed,
                    'pass': int(len(tests) > 0 and passed == len(tests)),
                    'error': error,
                    'extracted_code': code,
                }
            )

        detail = pd.DataFrame(rows)
        detail_file = get_intermediate_file_path(eval_file, '_lcbenchv6_eval')
        dump(detail, detail_file)

        ret = {
            'Overall': [round(float(detail['pass'].mean()) * 100, 2) if len(detail) else 0.0],
            'Correct': [int(detail['pass'].sum())],
            'Total': [int(len(detail))],
        }
        for diff in sorted(set(detail['difficulty']) - {''}):
            sub = detail[detail['difficulty'] == diff]
            ret[f'{diff}_acc'] = [round(float(sub['pass'].mean()) * 100, 2)]

        score = pd.DataFrame(ret)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(score, score_file)
        return score

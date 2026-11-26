from collections import defaultdict
import csv
from pathlib import Path

import numpy as np


def labeled_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'text', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: expected text and label columns.')
        rows = list(reader)
    texts = [row['text'] for row in rows]
    labels = [int(row['label']) for row in rows]
    if not texts or set(labels) != {0, 1}:
        raise ValueError(f'{path}: both labels 0 and 1 must be present.')
    if any(not text.strip() or len(text) > 8192 for text in texts):
        raise ValueError(f'{path}: text must contain 1–8,192 characters.')
    return rows


def read_split(path):
    return dataset(labeled_rows(path))



DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'data'
SOURCE_NAMES = ('HttpParamsDataset.csv', 'Mendeley_SQLi.csv', '외부_분할.csv')
EXTERNAL_EVALUATION = 'WAMM_v2_evaluation.csv'


def data_files(data_dir):
    return tuple(data_dir / name for name in SOURCE_NAMES)


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_partitions(data_dir):
    http, mendeley, split = data_files(data_dir)
    internal = defaultdict(list)
    for row in records(http):
        internal[row['split']].append(row)
    supplemental = defaultdict(list)
    split_by_id = {row['id']: row['split'] for row in records(split)}
    for row in records(mendeley):
        supplemental[split_by_id[row['id']]].append(row)
    for partitions in (internal, supplemental):
        seen = set()
        for rows in partitions.values():
            groups = {row['group'] for row in rows}
            if seen & groups:
                raise ValueError('Template groups overlap between data splits.')
            seen.update(groups)
    return internal, supplemental


def dataset(rows):
    return [row['text'] for row in rows], np.array([int(row['label']) for row in rows], dtype=np.int8)

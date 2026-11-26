from collections import defaultdict
import csv
from pathlib import Path

import numpy as np


def read_split(path):
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
    return texts, np.asarray(labels, dtype=np.int8)



DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'data'
HTTP_DATASET = 'HttpParamsDataset.csv'
EXTERNAL_DATASET = 'Mendeley_SQLi.csv'
EXTERNAL_SPLITS = '외부_분할.csv'


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def separate_groups(partitions):
    seen = set()
    for rows in partitions:
        groups = {row['group'] for row in rows}
        if seen & groups:
            raise ValueError('Template groups overlap between data splits.')
        seen.update(groups)


def load_internal_partitions(data_dir):
    internal = defaultdict(list)
    for row in records(data_dir / HTTP_DATASET):
        internal[row['split']].append(row)
    separate_groups(internal.values())
    return internal


def load_external_holdout(data_dir):
    ids = {row['id'] for row in records(data_dir / EXTERNAL_SPLITS) if row['split'] == 'holdout'}
    rows = [row for row in records(data_dir / EXTERNAL_DATASET) if row['id'] in ids]
    if not ids or len(rows) != len(ids) or {row['id'] for row in rows} != ids:
        raise ValueError('The external evaluation IDs do not match the dataset.')
    return rows


def dataset(rows):
    return [row['text'] for row in rows], np.array([int(row['label']) for row in rows], dtype=np.int8)

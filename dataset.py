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

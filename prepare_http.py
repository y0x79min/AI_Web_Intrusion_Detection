import argparse
import csv
import hashlib
import html
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote_plus

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def template(text):
    normal = unicodedata.normalize('NFKC', html.unescape(unquote_plus(text))).lower()
    normal = re.sub(r"'[^']*'", "'<str>'", normal)
    normal = re.sub(r'\b0x[0-9a-f]+\b', '<hex>', normal)
    normal = re.sub(r'\b\d+(?:\.\d+)?\b', '<num>', normal)
    return hashlib.sha256(re.sub(r'\s+', '', normal).encode('utf-8')).hexdigest()


def build(input_path, output_dir):
    grouped = {}
    with input_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'payload', 'attack_type'}.issubset(reader.fieldnames or []):
            raise ValueError('Expected HttpParamsDataset columns payload and attack_type.')
        for row in reader:
            kind, text = row['attack_type'], row['payload']
            if kind not in ('norm', 'sqli') or not text.strip() or len(text) > 8192:
                continue
            group = template(text)
            grouped.setdefault(group, []).append((text, int(kind == 'sqli')))
    records = [
        (text, items[0][1], group)
        for group, items in sorted(grouped.items()) if len({label for _, label in items}) == 1
        for text in sorted({text for text, _ in items}, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:3]
    ]
    labels = np.array([label for _, label, _ in records])
    groups = np.array([group for _, _, group in records])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_val, test = next(outer.split(labels, labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    train, validation = next(inner.split(labels[train_val], labels[train_val], groups[train_val]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in [('train', train_val[train]), ('validation', train_val[validation]), ('test', test)]:
        path = output_dir / f'{name}.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['text', 'label'])
            writer.writerows(records[index][:2] for index in indices)
        print(f'{name}: {len(indices):,} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare labeled HTTP input splits for offline experiments.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.input, args.output)

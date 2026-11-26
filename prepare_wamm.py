import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qsl

from dataset import records
from prepare_http import template

LABELS = {'000 - Normal': 0, '66 - SQL Injection': 1}
DEFAULT_PER_CLASS = 200
REQUEST_LINE = re.compile(r'GET ([^\s]+) HTTP/\d+(?:\.\d+)?')


def fingerprint(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def query_value(request):
    first_line = request.partition('\n')[0].rstrip('\r')
    match = REQUEST_LINE.fullmatch(first_line)
    if not match:
        return None
    target = match.group(1)
    if not target.startswith('/') or '?' not in target:
        return None
    query = target.partition('?')[2]
    try:
        fields = parse_qsl(query, keep_blank_values=True, errors='strict', max_num_fields=128)
    except (UnicodeError, ValueError):
        return None
    if len(fields) != 1:
        return None
    name, value = fields[0]
    if not name or not value.strip() or len(value) > 8192:
        return None
    return value


def build(database, data_dir, output, per_class):
    existing_groups = set()
    existing_texts = set()
    for name in ('HttpParamsDataset.csv', 'Mendeley_SQLi.csv'):
        for row in records(data_dir / name):
            value = row['text']
            existing_texts.add(fingerprint(value))
            existing_groups.add(template(value))

    counts = Counter()
    groups = {}
    conflicts = set()
    uri = database.resolve().as_uri() + '?mode=ro&immutable=1'
    with sqlite3.connect(uri, uri=True) as connection:
        for source_label, label in LABELS.items():
            cursor = connection.execute('SELECT col1 FROM combined_data WHERE col2 = ?', (source_label,))
            for (request,) in cursor:
                counts[label, 'source'] += 1
                value = query_value(request) if isinstance(request, str) else None
                if value is None:
                    continue
                counts[label, 'single_get_parameter'] += 1
                key = fingerprint(value)
                group = template(value)
                if key in existing_texts or group in existing_groups:
                    counts[label, 'development_overlap'] += 1
                    continue
                if group in conflicts:
                    continue
                previous = groups.get(group)
                if previous is not None and previous[0] != label:
                    conflicts.add(group)
                    del groups[group]
                elif previous is None or key < previous[1]:
                    groups[group] = (label, key, value)

    eligible = {label: sorted((group, key, value) for group, (y, key, value) in groups.items() if y == label)
                for label in LABELS.values()}
    if any(len(values) < per_class for values in eligible.values()):
        raise ValueError(f'Need {per_class} independent templates per class; found '
                         + str({label: len(values) for label, values in eligible.items()}))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('id', 'text', 'label'))
        for label, values in eligible.items():
            for _, key, value in values[:per_class]:
                writer.writerow((key, value, label))
    return {
        'sampling_unit': 'GET requests with one nonblank query parameter and a published request label',
        'sampling_rule': 'One value per normalized template, remove development overlaps and label conflicts, '
                         'then take the first per-class templates in SHA-256 order.',
        'source_counts': {label: counts[label, 'source'] for label in LABELS.values()},
        'single_parameter_counts': {label: counts[label, 'single_get_parameter'] for label in LABELS.values()},
        'excluded_development_overlap': {label: counts[label, 'development_overlap'] for label in LABELS.values()},
        'excluded_conflicting_templates': len(conflicts),
        'eligible_templates': {label: len(values) for label, values in eligible.items()},
        'sample_counts': {label: per_class for label in LABELS.values()},
        'sample_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare a fixed offline external evaluation from WAMM v2.')
    parser.add_argument('database', type=Path)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-class', type=int, default=DEFAULT_PER_CLASS)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error('--per-class must be positive.')
    stats = build(args.database, args.data_dir, args.output, args.per_class)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

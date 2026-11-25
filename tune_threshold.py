import argparse
from bisect import bisect_left
import csv
from fractions import Fraction
import json
from pathlib import Path

REFERENCE_THRESHOLD = 0.5
GRID_SIZE = 1000


def read_validation(path):
    grouped = {}
    required = {'source', 'split', 'label', 'score'}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream)
        if not required.issubset(rows.fieldnames or []):
            raise ValueError('Expected source, split, label, and score columns.')
        for row in rows:
            if row['split'] != 'validation':
                raise ValueError('Only validation data can be used to choose a threshold.')
            label, score = int(row['label']), float(row['score'])
            if label not in (0, 1) or not 0 <= score <= 1 or not row['source'].strip():
                raise ValueError('Invalid source, binary label, or score.')
            grouped.setdefault(row['source'], []).append((score, label))
    return grouped


def score_rows(source, split, rows, labels, scores):
    for row, label, score in zip(rows, labels, scores, strict=True):
        sample_id = row.get('sample_id') or row['id']
        yield source, split, sample_id, int(label), float(score)


def write_scores(path, rows, threshold):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('source', 'split', 'sample_id', 'label', 'score', 'threshold', 'prediction'))
        writer.writerows((*row, threshold, int(row[-1] >= threshold)) for row in rows)
    temporary.replace(path)


def select_threshold(groups):
    if not groups:
        raise ValueError('No validation rows provided.')
    ranked = {}
    for source, rows in groups.items():
        negative = sorted(score for score, label in rows if label == 0)
        positive = sorted(score for score, label in rows if label == 1)
        if not negative or not positive:
            raise ValueError(f'{source}: both classes must appear.')
        ranked[source] = negative, positive

    def summary(threshold):
        details = {}
        total_accuracy = total_recall = Fraction(0)
        for source, (normal, attack) in ranked.items():
            tn = bisect_left(normal, threshold)
            fn = bisect_left(attack, threshold)
            fp, tp = len(normal) - tn, len(attack) - fn
            recall = Fraction(tp, len(attack))
            balanced = (recall + Fraction(tn, len(normal))) / 2
            total_accuracy += balanced
            total_recall += recall
            details[source] = {'n': len(normal) + len(attack), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                               'recall': float(recall), 'balanced_accuracy': float(balanced)}
        return total_accuracy / len(ranked), total_recall / len(ranked), details

    best_rank = None
    for step in range(GRID_SIZE + 1):
        threshold = step / GRID_SIZE
        balanced, recall, details = summary(threshold)
        rank = balanced, recall, -abs(step - GRID_SIZE // 2)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            selected = {'threshold': threshold, 'mean_balanced_accuracy': float(balanced),
                        'mean_recall': float(recall), 'sources': details}
    examples = []
    for threshold in (0.4, 0.5, 0.6):
        balanced, recall, details = summary(threshold)
        examples.append({'threshold': threshold, 'mean_balanced_accuracy': float(balanced),
                         'mean_recall': float(recall), 'sources': details})
    balanced, recall, details = summary(REFERENCE_THRESHOLD)
    return {
        'method': 'Maximize mean balanced accuracy across validation sources; ties prefer mean attack recall, then the closest value to 0.5.',
        'grid_step': 1 / GRID_SIZE,
        'reference': {'threshold': REFERENCE_THRESHOLD, 'mean_balanced_accuracy': float(balanced),
                      'mean_recall': float(recall), 'sources': details},
        'selected': selected,
        'examples': examples,
    }


def main():
    parser = argparse.ArgumentParser(description='Choose a score cutoff using labeled validation scores.')
    parser.add_argument('validation_csv', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = select_threshold(read_validation(args.validation_csv))
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()

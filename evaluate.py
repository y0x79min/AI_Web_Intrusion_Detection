import argparse
import json
from pathlib import Path

from tensorflow import keras
from sklearn.metrics import confusion_matrix

from dataset import DEFAULT_DATA_DIR, data_files, dataset, load_partitions, read_split
from detector import MODEL_DIR, MODEL_FILE, THRESHOLD, byte_scores, digest, load_byte_model, read_selection


def save_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def measure(labels, values):
    pred = values >= THRESHOLD
    tn, fp, fn, tp = [int(n) for n in confusion_matrix(labels, pred, labels=[0, 1]).ravel()]
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {'n': len(labels), 'accuracy': (tn + tp) / len(labels),
            'f1': 2 * tp / (2 * tp + fp + fn), 'recall': recall,
            'specificity': specificity, 'balanced_accuracy': (recall + specificity) / 2,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def evaluate(data_dir, artifact_dir=MODEL_DIR):
    protocol = read_selection(artifact_dir)
    actual_hashes = {path.name: digest(path) for path in data_files(data_dir)}
    if actual_hashes != protocol['source_data_sha256']:
        raise ValueError('The data files do not match this model’s evaluation split.')
    internal, external = load_partitions(data_dir)
    model = load_byte_model(artifact_dir)
    max_bytes = protocol.get('max_bytes')
    splits = {'train': internal['train'], 'validation': internal['validation'],
              'test': internal['test'], 'external_test': external['holdout']}
    results = {}
    for name, rows in splits.items():
        texts, labels = dataset(rows)
        results[name] = measure(labels, byte_scores(model, texts, max_bytes=max_bytes))
    return {
        'model': MODEL_FILE, 'model_sha256': protocol['model_sha256'],
        'threshold': THRESHOLD, 'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved byte CNN.')
    parser.add_argument('csv', type=Path, nargs='?', help='Evaluate a single CSV with text and label columns.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--artifact-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--model-file', type=Path, help='Use a specified model with a single CSV.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.csv:
        texts, labels = read_split(args.csv)
        if args.model_file:
            model = keras.models.load_model(args.model_file, compile=False)
            max_bytes = None
        else:
            model = load_byte_model(args.artifact_dir)
            max_bytes = read_selection(args.artifact_dir).get('max_bytes')
        metrics = measure(labels, byte_scores(model, texts, max_bytes=max_bytes))
        positive = metrics['tp'] + metrics['fp']
        precision = metrics['tp'] / positive if positive else 0.0
        report = {'samples': metrics['n'], 'threshold': THRESHOLD, 'precision': precision, **metrics}
    else:
        if args.model_file:
            parser.error('--model-file requires a CSV path.')
        report = evaluate(args.data_dir, args.artifact_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

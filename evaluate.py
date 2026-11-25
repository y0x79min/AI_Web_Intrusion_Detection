import argparse
import json
from pathlib import Path

from tensorflow import keras
from sklearn.metrics import confusion_matrix

from dataset import DEFAULT_DATA_DIR, data_files, dataset, load_partitions, read_split
from detector import MODEL_DIR, MODEL_FILE, THRESHOLD, byte_scores, digest, load_byte_model, read_selection
from tune_threshold import score_rows, write_scores


def save_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def measure(labels, values, threshold=THRESHOLD):
    pred = values.astype('float64') >= threshold
    tn, fp, fn, tp = [int(n) for n in confusion_matrix(labels, pred, labels=[0, 1]).ravel()]
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {'n': len(labels), 'accuracy': (tn + tp) / len(labels),
            'f1': 2 * tp / (2 * tp + fp + fn), 'recall': recall,
            'specificity': specificity, 'balanced_accuracy': (recall + specificity) / 2,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def evaluate(data_dir, artifact_dir=MODEL_DIR, *, scores_dir=None, validation_csv=None):
    protocol = read_selection(artifact_dir)
    actual_hashes = {path.name: digest(path) for path in data_files(data_dir)}
    if actual_hashes != protocol['source_data_sha256']:
        raise ValueError('The data files do not match this model’s evaluation split.')
    internal, external = load_partitions(data_dir)
    model = load_byte_model(artifact_dir)
    max_bytes = protocol.get('max_bytes')
    threshold = protocol['threshold']
    if scores_dir is not None or validation_csv is not None:
        validation = []
        for source, rows in (('http', internal['validation']), ('mendeley', external['validation'])):
            texts, labels = dataset(rows)
            scores = byte_scores(model, texts, max_bytes=max_bytes)
            validation.extend(score_rows(source, 'validation', rows, labels, scores))
        write_scores(validation_csv or scores_dir / 'validation_scores.csv', validation, threshold)
        if validation_csv is not None:
            return None

    evaluation_rows = []
    splits = {'train': internal['train'], 'validation': internal['validation'],
              'test': internal['test'], 'external_test': external['holdout']}
    results = {}
    for name, rows in splits.items():
        texts, labels = dataset(rows)
        scores = byte_scores(model, texts, max_bytes=max_bytes)
        results[name] = measure(labels, scores, threshold)
        if scores_dir is not None and name in ('test', 'external_test'):
            source = 'http' if name == 'test' else 'mendeley'
            evaluation_rows.extend(score_rows(source, name, rows, labels, scores))
    if scores_dir is not None:
        write_scores(scores_dir / 'test_scores.csv', evaluation_rows, threshold)
    return {
        'model': MODEL_FILE, 'model_sha256': protocol['model_sha256'],
        'threshold': threshold, 'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved byte CNN.')
    parser.add_argument('csv', type=Path, nargs='?', help='Evaluate a single CSV with text and label columns.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--artifact-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--model-file', type=Path, help='Use a specified model with a single CSV.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scores-dir', type=Path, help='Export validation and test decisions as CSV files.')
    parser.add_argument('--validation-csv', type=Path, help='Export validation scores only.')
    args = parser.parse_args()
    if args.csv and (args.scores_dir or args.validation_csv):
        parser.error('CSV score exports require the prepared data directory.')
    if args.validation_csv and (args.scores_dir or args.output):
        parser.error('--validation-csv is a separate validation-only action.')
    if args.csv:
        texts, labels = read_split(args.csv)
        if args.model_file:
            model = keras.models.load_model(args.model_file, compile=False)
            max_bytes, threshold = None, THRESHOLD
        else:
            model = load_byte_model(args.artifact_dir)
            protocol = read_selection(args.artifact_dir)
            max_bytes, threshold = protocol.get('max_bytes'), protocol['threshold']
        metrics = measure(labels, byte_scores(model, texts, max_bytes=max_bytes), threshold)
        positive = metrics['tp'] + metrics['fp']
        precision = metrics['tp'] / positive if positive else 0.0
        report = {'samples': metrics['n'], 'threshold': threshold, 'precision': precision, **metrics}
    else:
        if args.model_file:
            parser.error('--model-file requires a CSV path.')
        report = evaluate(args.data_dir, args.artifact_dir,
                          scores_dir=args.scores_dir, validation_csv=args.validation_csv)
    if report is None:
        return
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

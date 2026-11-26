import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import numpy as np

from dataset import (DEFAULT_DATA_DIR, EXTERNAL_DATASET, EXTERNAL_SPLITS, HTTP_DATASET, dataset,
                     load_external_holdout, load_internal_partitions)
from detector import ROOT, THRESHOLD, byte_scores, digest, load_byte_model, read_selection
from evaluate import measure, save_json
from train import BATCH_SIZE, EPOCHS, OPTIMIZERS, SEED

RATES = (0.001, 0.003, 0.01)


def case_name(optimizer, rate):
    return f'{optimizer}_{rate:g}'


def fit_case(data_dir, output_dir, optimizer, rate):
    directory = output_dir / case_name(optimizer, rate)
    if (directory / 'byte_selection.json').is_file():
        return
    print(f'{directory.name}: training', flush=True)
    with (output_dir / f'{directory.name}.log').open('x', encoding='utf-8') as log:
        subprocess.run(
            [sys.executable, '-X', 'utf8', str(ROOT / 'train.py'), '--data-dir', str(data_dir),
             '--output-dir', str(directory), '--optimizer', optimizer, '--learning-rate', str(rate)],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    print(f'{directory.name}: fitted', flush=True)


def validate(model, texts, labels):
    scores = byte_scores(model, texts).astype(np.float64)
    result = measure(labels, scores, THRESHOLD)
    tp, tn, fp, fn = (result[key] for key in ('tp', 'tn', 'fp', 'fn'))
    recall = Fraction(tp, tp + fn)
    balanced = (recall + Fraction(tn, tn + fp)) / 2
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    loss = float(np.mean(-labels * np.log(scores) - (1 - labels) * np.log1p(-scores)))
    return balanced, recall, loss


def compare(data_dir, output_dir, jobs, external_dir=None):
    combinations = [(opt, rate) for opt in OPTIMIZERS for rate in RATES]
    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        tasks = [executor.submit(fit_case, data_dir, output_dir, opt, rate) for opt, rate in combinations]
        for task in tasks:
            task.result()

    hashes = {HTTP_DATASET: digest(data_dir / HTTP_DATASET)}
    validation_text, validation_labels = dataset(load_internal_partitions(data_dir)['validation'])
    checkpoints = []
    for opt, rate in combinations:
        directory = output_dir / case_name(opt, rate)
        selection = read_selection(directory)
        settings = (selection['optimizer'], selection['learning_rate'], selection['max_bytes'],
                    selection['epoch_budget'], selection['batch_size'], selection['seed'])
        if settings != (opt, rate, None, EPOCHS, BATCH_SIZE, SEED) or selection['source_data_sha256'] != hashes:
            raise ValueError(f'Unexpected experiment settings: {directory}')
        model = load_byte_model(directory)
        balanced, recall, loss = validate(model, validation_text, validation_labels)
        summary = {'optimizer': opt, 'learning_rate': rate, 'epoch': selection['epoch'],
                   'model_sha256': selection['model_sha256'], 'validation_balanced_accuracy': float(balanced),
                   'validation_recall': float(recall), 'validation_log_loss': loss}
        checkpoints.append(((balanced, recall, -loss), summary, directory, selection['threshold']))

    _, winner, winner_path, threshold = max(checkpoints, key=lambda checkpoint: checkpoint[0])
    frozen = {'selection_rule': 'At cutoff 0.5, maximize HttpParamsDataset validation balanced accuracy, then attack recall, then minimize log loss.',
              'selected': winner, 'threshold': threshold, 'candidate_count': len(checkpoints),
              'source_data_sha256': hashes}
    save_json(output_dir / 'winner.json', frozen)
    print(f'Validation winner: {winner_path.name}, epoch {winner["epoch"]}, threshold {threshold:.3f}', flush=True)

    external_dir = external_dir or data_dir
    external_hashes = {name: digest(external_dir / name) for name in (EXTERNAL_DATASET, EXTERNAL_SPLITS)}
    texts, labels = dataset(load_external_holdout(external_dir))
    results = []
    for _, summary, directory, _ in checkpoints:
        model = load_byte_model(directory)
        metrics = measure(labels, byte_scores(model, texts), threshold)
        results.append({**summary, 'selected': directory == winner_path, **metrics})
        print(f'{directory.name}: {metrics["accuracy"]:.2%}, FP={metrics["fp"]}, FN={metrics["fn"]}', flush=True)
    save_json(output_dir / 'comparison.json', {**frozen, 'results': results, 'external_data_sha256': external_hashes})


def main():
    parser = argparse.ArgumentParser(description='Compare optimizers and learning rates on full UTF-8 inputs.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--external-data-dir', type=Path, help='Read the external holdout here after model selection.')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive.')
    compare(args.data_dir, args.output_dir, args.jobs, args.external_data_dir)


if __name__ == '__main__':
    main()

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import numpy as np

from dataset import DEFAULT_DATA_DIR, EXTERNAL_EVALUATION, data_files, dataset, load_partitions, read_split
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


def validate(model, splits):
    balanced = recall = Fraction(0)
    losses = []
    for texts, labels in splits:
        scores = byte_scores(model, texts).astype(np.float64)
        result = measure(labels, scores, THRESHOLD)
        tp, tn, fp, fn = (result[key] for key in ('tp', 'tn', 'fp', 'fn'))
        hit_rate = Fraction(tp, tp + fn)
        balanced += (hit_rate + Fraction(tn, tn + fp)) / 2
        recall += hit_rate
        scores = np.clip(scores, 1e-7, 1 - 1e-7)
        losses.append(float(np.mean(-labels * np.log(scores) - (1 - labels) * np.log1p(-scores))))
    return balanced / len(splits), recall / len(splits), float(np.mean(losses))


def compare(data_dir, output_dir, jobs, external_csv):
    combinations = [(opt, rate) for opt in OPTIMIZERS for rate in RATES]
    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        tasks = [executor.submit(fit_case, data_dir, output_dir, opt, rate) for opt, rate in combinations]
        for task in tasks:
            task.result()

    hashes = {path.name: digest(path) for path in data_files(data_dir)}
    internal, supplemental = load_partitions(data_dir)
    splits = [dataset(source['validation']) for source in (internal, supplemental)]
    checkpoints = []
    for opt, rate in combinations:
        directory = output_dir / case_name(opt, rate)
        selection = read_selection(directory)
        settings = (selection['optimizer'], selection['learning_rate'], selection['max_bytes'],
                    selection['epoch_budget'], selection['batch_size'], selection['seed'])
        if settings != (opt, rate, None, EPOCHS, BATCH_SIZE, SEED) or selection['source_data_sha256'] != hashes:
            raise ValueError(f'Unexpected experiment settings: {directory}')
        model = load_byte_model(directory)
        balanced, recall, loss = validate(model, splits)
        summary = {'optimizer': opt, 'learning_rate': rate, 'epoch': selection['epoch'],
                   'model_sha256': selection['model_sha256'], 'validation_balanced_accuracy': float(balanced),
                   'validation_recall': float(recall), 'validation_log_loss': loss}
        checkpoints.append(((balanced, recall, -loss), summary, directory, selection['threshold']))

    _, winner, winner_path, threshold = max(checkpoints, key=lambda checkpoint: checkpoint[0])
    frozen = {'selection_rule': 'At cutoff 0.5, maximize mean validation balanced accuracy, then mean recall, then minimize mean log loss.',
              'selected': winner, 'threshold': threshold, 'candidate_count': len(checkpoints),
              'source_data_sha256': hashes}
    save_json(output_dir / 'winner.json', frozen)
    print(f'Validation winner: {winner_path.name}, epoch {winner["epoch"]}, threshold {threshold:.3f}', flush=True)

    external_hash = digest(external_csv)
    texts, labels = read_split(external_csv)
    results = []
    for _, summary, directory, _ in checkpoints:
        model = load_byte_model(directory)
        metrics = measure(labels, byte_scores(model, texts), threshold)
        results.append({**summary, 'selected': directory == winner_path, **metrics})
        print(f'{directory.name}: {metrics["accuracy"]:.2%}, FP={metrics["fp"]}, FN={metrics["fn"]}', flush=True)
    save_json(output_dir / 'comparison.json', {**frozen, 'results': results, 'external_data_sha256': {external_csv.name: external_hash}})


def main():
    parser = argparse.ArgumentParser(description='Compare optimizers and learning rates on full UTF-8 inputs.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    external = parser.add_mutually_exclusive_group()
    external.add_argument('--external-csv', type=Path, help='Prepared WAMM evaluation CSV.')
    external.add_argument('--external-data-dir', type=Path, help='Directory containing the prepared WAMM CSV.')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive.')
    external_csv = args.external_csv or (args.external_data_dir or args.data_dir) / EXTERNAL_EVALUATION
    compare(args.data_dir, args.output_dir, args.jobs, external_csv)


if __name__ == '__main__':
    main()

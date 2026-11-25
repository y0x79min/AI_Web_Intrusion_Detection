import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from dataset import DEFAULT_DATA_DIR, data_files, dataset, load_partitions
from detector import MODEL_DIR, ROOT, byte_scores, digest, load_byte_model, read_selection
from evaluate import measure, save_json

LIMITS = (128, 256, None)
RATES = (0.0003, 0.001, 0.003)
BASELINE = (None, 0.001)


def case_name(max_bytes, learning_rate):
    return f'{max_bytes or "full"}_{learning_rate:g}'


def fit_case(data_dir, output_dir, limit, rate):
    directory = output_dir / case_name(limit, rate)
    if (directory / 'byte_selection.json').is_file():
        return
    command = [sys.executable, '-X', 'utf8', str(ROOT / 'train.py'), '--data-dir', str(data_dir),
               '--output-dir', str(directory), '--learning-rate', str(rate)]
    if limit is not None:
        command += ['--max-bytes', str(limit)]
    print(f'{directory.name}: training', flush=True)
    with (output_dir / f'{directory.name}.log').open('x', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'{directory.name}: fitted', flush=True)


def compare(data_dir, output_dir, jobs):
    combinations = [(limit, rate) for limit in LIMITS for rate in RATES]
    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        tasks = [executor.submit(fit_case, data_dir, output_dir, limit, rate)
                 for limit, rate in combinations if (limit, rate) != BASELINE]
        for task in tasks:
            task.result()

    hashes = {path.name: digest(path) for path in data_files(data_dir)}
    _, external = load_partitions(data_dir)
    texts, labels = dataset(external['holdout'])
    threshold = read_selection(MODEL_DIR)['threshold']
    results = []
    for limit, rate in combinations:
        baseline = (limit, rate) == BASELINE
        directory = MODEL_DIR if baseline else output_dir / case_name(limit, rate)
        selection = read_selection(directory)
        if (selection['source_data_sha256'] != hashes or selection.get('max_bytes') != limit or
                selection['learning_rate'] != rate or selection['epoch_budget'] != 6 or
                selection['batch_size'] != 128 or selection['seed'] != 42):
            raise ValueError(f'Comparison settings differ: {directory}')
        model = load_byte_model(directory)
        metrics = measure(labels, byte_scores(model, texts, max_bytes=limit), threshold)
        results.append({'max_bytes': limit, 'learning_rate': rate, 'baseline': baseline,
                        'epoch': selection['epoch'], 'model_sha256': selection['model_sha256'], **metrics})
        print(f'{case_name(limit, rate)}: {metrics["accuracy"]:.2%}, '
              f'FP={metrics["fp"]}, FN={metrics["fn"]}', flush=True)
    save_json(output_dir / 'comparison.json', {
        'threshold': threshold, 'results': results, 'source_data_sha256': hashes,
    })


def main():
    parser = argparse.ArgumentParser(description='Compare byte limits and learning rates on the same external split.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive.')
    compare(args.data_dir, args.output_dir, args.jobs)


if __name__ == '__main__':
    main()

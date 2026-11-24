import argparse
import json
from pathlib import Path

from detector import Encoder, MODELS_BY_NAME, load_model, score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from dataset import read_split


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved CNN on a labeled CSV.')
    parser.add_argument('csv', type=Path)
    parser.add_argument('--model', choices=MODELS_BY_NAME, default='improved')
    parser.add_argument('--model-file', type=Path)
    args = parser.parse_args()
    texts, labels = read_split(args.csv)
    values = score(load_model(args.model_file or MODELS_BY_NAME[args.model]), Encoder(), texts)
    predicted = values >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average='binary', zero_division=0)
    print(json.dumps({
        'samples': len(texts), 'threshold': 0.5, 'accuracy': float(accuracy_score(labels, predicted)),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

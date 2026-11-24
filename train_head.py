import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.metrics import f1_score

from dataset import read_split
from detector import Encoder, MODELS_BY_NAME, clean_text, load_model, score


def train(train_path, validation_path, output, epochs):
    train_text, train_labels = read_split(train_path)
    val_text, val_labels = read_split(validation_path)
    overlap = set(map(clean_text, train_text)).intersection(map(clean_text, val_text))
    if overlap:
        raise ValueError('Training and validation contain overlapping cleaned text.')
    if output.exists() or output.suffix.lower() != '.h5' or epochs < 1:
        raise ValueError('Choose a new .h5 output file and at least one epoch.')

    keras.utils.set_random_seed(42)
    model = load_model(MODELS_BY_NAME['baseline'])
    for layer in model.layers:
        layer.trainable = isinstance(layer, keras.layers.Dense)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
    encoder = Encoder()
    rng = np.random.default_rng(42)
    best_f1 = -1.0
    best_weights = None
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(train_text))
        model.reset_metrics()
        for start in range(0, len(indices), 128):
            batch = indices[start:start + 128]
            inputs = encoder.transform([train_text[index] for index in batch])
            loss = model.train_on_batch(inputs, train_labels[batch])
        predicted = score(model, encoder, val_text) >= 0.5
        f1 = f1_score(val_labels, predicted, zero_division=0)
        print(f'epoch={epoch} loss={float(loss):.4f} validation_f1={f1:.4f}', flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = model.get_weights()

    model.set_weights(best_weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output, include_optimizer=False)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the saved CNN’s Dense layers.')
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--validation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--epochs', default=4, type=int)
    args = parser.parse_args()
    train(args.train, args.validation, args.output, args.epochs)


if __name__ == '__main__':
    main()

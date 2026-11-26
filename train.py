import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import DEFAULT_DATA_DIR, HTTP_DATASET, dataset, load_internal_partitions
from detector import MODEL_FILE, SELECTION_FILE, THRESHOLD, byte_scores, digest, encode_bytes
from evaluate import measure, save_json
from tune_threshold import select_threshold

SEED = 42
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.003
OPTIMIZER = 'rmsprop'
OPTIMIZERS = ('adam', 'rmsprop', 'sgd')
SGD_MOMENTUM = 0.9


def make_optimizer(name, learning_rate):
    if name == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=SGD_MOMENTUM)
    optimizers = {'adam': keras.optimizers.Adam, 'rmsprop': keras.optimizers.RMSprop}
    return optimizers[name](learning_rate=learning_rate)


def make_byte_cnn(learning_rate, optimizer=OPTIMIZER):
    inputs = keras.Input(shape=(None,), dtype='int32', name='utf8_byte_ids')
    x = keras.layers.Embedding(257, 12)(inputs)
    x = keras.layers.Conv1D(20, 5, padding='same', activation='relu')(x)
    x = keras.layers.Conv1D(24, 7, padding='same', activation='relu')(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(24, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=make_optimizer(optimizer, learning_rate), loss='binary_crossentropy')
    return model


def fit(data_dir, output_dir, *, learning_rate=LEARNING_RATE, optimizer=OPTIMIZER, max_bytes=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MODEL_FILE
    selected = output_dir / SELECTION_FILE
    for artifact in (path, selected, output_dir / 'byte_history.json'):
        if artifact.exists():
            raise FileExistsError(f'Choose an empty training output folder: {artifact}')
    source = data_dir / HTTP_DATASET
    source_hash = digest(source)
    internal = load_internal_partitions(data_dir)
    train_text, train_labels = dataset(internal['train'])
    validation_text, validation_labels = dataset(internal['validation'])

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = make_byte_cnn(learning_rate, optimizer)
    history = []
    best_rank = None
    best_weights = None
    best_record = None
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_text))
        model.reset_metrics()
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            features = encode_bytes([train_text[i] for i in batch], max_bytes)
            loss = model.train_on_batch(features, train_labels[batch])

        metrics = measure(validation_labels, byte_scores(model, validation_text, max_bytes=max_bytes))
        rank = (metrics['balanced_accuracy'], metrics['recall'])
        item = {'epoch': epoch, 'loss': float(loss), 'validation': {'http': metrics}, 'selection_score': rank[0]}
        history.append(item)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_weights = model.get_weights()
            best_record = item
        print(f'Epoch {epoch}: loss={float(loss):.4f} '
              f'best validation balanced accuracy={best_rank[0]:.4f}', flush=True)

    if digest(source) != source_hash:
        raise ValueError('The training data changed during training.')
    model.set_weights(best_weights)
    model.save(path, include_optimizer=False)
    reload = keras.models.load_model(path, compile=False)
    texts = validation_text[:32]
    np.testing.assert_allclose(byte_scores(model, texts, max_bytes=max_bytes),
                               byte_scores(reload, texts, max_bytes=max_bytes), atol=1e-7)
    validation_scores = byte_scores(reload, validation_text, max_bytes=max_bytes)
    threshold_report = select_threshold({
        'http': list(zip(map(float, validation_scores), map(int, validation_labels), strict=True))
    })
    threshold = threshold_report['selected']['threshold']
    save_json(output_dir / 'threshold_selection.json', threshold_report)
    save_json(output_dir / 'byte_history.json', history)
    save_json(selected, {
        **best_record, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'model': path.name, 'model_sha256': digest(path),
        'source_data_sha256': {HTTP_DATASET: source_hash},
        'threshold': threshold, 'checkpoint_threshold': THRESHOLD,
        'threshold_from': 'http_validation_balanced_accuracy', 'seed': SEED, 'epoch_budget': EPOCHS,
        'optimizer': optimizer, 'optimizer_config': model.optimizer.get_config(),
        'learning_rate': learning_rate, 'batch_size': BATCH_SIZE, 'max_bytes': max_bytes,
        'train_http_n': len(train_text),
        'selection': 'HttpParamsDataset validation balanced accuracy; ties prefer higher attack recall, then the earlier epoch.',
    })
    print(f'Selected epoch {best_record["epoch"]} using validation.', flush=True)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    parser = argparse.ArgumentParser(description='Train a byte CNN on the prepared splits.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--optimizer', choices=OPTIMIZERS, default=OPTIMIZER)
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--max-bytes', type=int)
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be a positive finite number.')
    if args.max_bytes is not None and args.max_bytes < 1:
        parser.error('--max-bytes must be positive.')
    fit(args.data_dir, args.output_dir, learning_rate=args.learning_rate,
        optimizer=args.optimizer, max_bytes=args.max_bytes)


if __name__ == '__main__':
    main()

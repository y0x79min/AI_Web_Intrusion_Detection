import hashlib
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
MODEL_FILE = 'byte_cnn.h5'
SELECTION_FILE = 'byte_selection.json'
THRESHOLD = 0.5
BYTE_PADDING = 12


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode_bytes(texts, max_bytes=None):
    raw = [text.encode('utf-8')[:max_bytes] for text in texts]
    values = np.zeros((len(raw), max(map(len, raw), default=0) + BYTE_PADDING), dtype=np.int32)
    for index, text in enumerate(raw):
        values[index, :len(text)] = np.frombuffer(text, dtype=np.uint8).astype(np.int32) + 1
    return values


def byte_scores(model, texts, batch_size=128, *, max_bytes=None):
    if not texts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate([
        model(encode_bytes(texts[i:i + batch_size], max_bytes), training=False).numpy().ravel()
        for i in range(0, len(texts), batch_size)
    ])


def read_selection(directory=MODEL_DIR):
    return json.loads((directory / SELECTION_FILE).read_text(encoding='utf-8'))


def load_byte_model(directory=MODEL_DIR):
    protocol = read_selection(directory)
    if protocol['model'] != MODEL_FILE or protocol['threshold'] != THRESHOLD:
        raise ValueError('The saved model settings do not match the detector.')
    path = directory / MODEL_FILE
    if digest(path) != protocol['model_sha256']:
        raise ValueError('The byte CNN checkpoint does not match the evaluated file.')
    return keras.models.load_model(path, compile=False)

import hashlib
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
INPUT_SHAPE = (64, 64, 1)
MODELS_BY_NAME = {
    'baseline': MODELS / 'baseline_cnn.h5',
    'improved': MODELS / 'improved_cnn.h5',
}
REPLACEMENTS = (
    ('\n', ''), ('%20', ' '), ('=', ' = '), ('((', ' (( '), ('))', ' )) '),
    ('(', ' ( '), (')', ' ) '), ('1 ', 'numeric'), (' 1', 'numeric'),
    ("'1 ", "'numeric "), (" 1'", " numeric'"), ('1,', 'numeric,'),
    (' 2 ', ' numeric '), (' 3 ', ' numeric '), (' 3--', ' numeric--'),
    (' 4 ', ' numeric '), (' 5 ', ' numeric '), (' 6 ', ' numeric '),
    (' 7 ', ' numeric '), (' 8 ', ' numeric '), ('1234', ' numeric '),
    ('22', ' numeric '), (' 8 ', ' numeric '), (' 200 ', ' numeric '),
    ('23 ', ' numeric '), ('"1', '"numeric'), ('1"', '"numeric'),
    ('7659', 'numeric'), (' 37 ', ' numeric '), (' 45 ', ' numeric '),
)


def clean_text(text):
    for source, target in REPLACEMENTS:
        text = text.replace(source, target)
    return text


class Encoder:
    def __init__(self, spec_path=MODELS / 'count_vectorizer.json'):
        spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
        options = spec['options']
        options['ngram_range'] = tuple(options['ngram_range'])
        self.analyze = CountVectorizer(**options).build_analyzer()
        self.positions = spec['token_sha256_to_index']
        self.width = int(np.prod(INPUT_SHAPE))
        if sorted(self.positions.values()) != list(range(self.width)):
            raise ValueError('The vocabulary does not match the CNN input dimensions.')

    def transform(self, texts):
        values = np.zeros((len(texts), self.width), dtype=np.float32)
        for row, text in zip(values, texts, strict=True):
            for token in self.analyze(clean_text(text)):
                digest = hashlib.sha256(token.encode('utf-8')).hexdigest()
                position = self.positions.get(digest)
                if position is not None:
                    row[position] += 1
        return values.reshape(-1, *INPUT_SHAPE)


def load_model(path):
    model = keras.models.load_model(path, compile=False)
    if tuple(model.input_shape[1:]) != INPUT_SHAPE or model.output_shape[-1] != 1:
        raise ValueError('Unexpected model input or output dimensions.')
    return model


def score(model, encoder, texts, batch_size=128):
    if not texts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate([
        model(encoder.transform(texts[start:start + batch_size]), training=False).numpy().ravel()
        for start in range(0, len(texts), batch_size)
    ])

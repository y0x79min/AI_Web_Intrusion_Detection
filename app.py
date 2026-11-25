from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from flask import Flask, abort, jsonify, request, send_from_directory

from detector import ROOT, byte_scores, load_byte_model

SCHEMA = '''
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, input TEXT NOT NULL, score REAL NOT NULL
)
'''


def create_app(database=None, scorer=None):
    app = Flask(__name__, static_folder=str(ROOT / 'web'), static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
    app.config['TRUSTED_HOSTS'] = ['127.0.0.1', 'localhost']
    db = Path(database or ROOT / 'instance' / 'demo.sqlite3')
    db.parent.mkdir(parents=True, exist_ok=True)

    def connect():
        connection = sqlite3.connect(db)
        connection.row_factory = sqlite3.Row
        return connection

    with closing(connect()) as connection:
        connection.execute(SCHEMA)
        connection.commit()

    if scorer is None:
        model = load_byte_model()

        def scorer(text):
            return float(byte_scores(model, [text])[0])

    def read_input():
        if not request.is_json:
            abort(415)
        body = request.get_json()
        text = body.get('input') if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            abort(400, 'input must be a string of 1–8,192 characters')
        return text

    @app.get('/')
    def web_page():
        return send_from_directory(app.static_folder, 'index.html')

    @app.get('/logs')
    def log_page():
        return send_from_directory(app.static_folder, 'logs.html')

    @app.post('/ai')
    def analyze():
        return jsonify(score=scorer(read_input()))

    @app.post('/demo/request')
    def submit_demo():
        text = read_input()
        value = scorer(text)
        timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with closing(connect()) as connection:
            row = connection.execute(
                'INSERT INTO request_log (at, input, score) VALUES (?, ?, ?)',
                (timestamp, text, value),
            )
            connection.commit()
        return jsonify(id=row.lastrowid, score=value, flagged=round(value, 2) > 0.6), 201

    @app.get('/data')
    def list_logs():
        with closing(connect()) as connection:
            rows = connection.execute(
                'SELECT id, at, score FROM request_log ORDER BY id DESC LIMIT 100'
            ).fetchall()
        return jsonify([dict(row) | {'flagged': round(row['score'], 2) > 0.6} for row in rows])

    @app.get('/data/<int:log_id>')
    def get_log(log_id):
        with closing(connect()) as connection:
            row = connection.execute(
                'SELECT id, at, input, score FROM request_log WHERE id = ?', (log_id,)
            ).fetchone()
        if row is None:
            abort(404)
        return jsonify(dict(row))

    return app


if __name__ == '__main__':
    create_app().run(
        host='127.0.0.1', port=8000, debug=False,
    )

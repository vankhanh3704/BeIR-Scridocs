import os

from flask import Flask, render_template, request, jsonify
from search import (
    search_tfidf, search_boolean,
    search_lsa, search_bm25,
    search_all, get_stats
)

app = Flask(__name__)
MAX_TOP_K = 50
VALID_METHODS = {'tfidf', 'boolean', 'lsa', 'bm25', 'all'}
VALID_BOOLEAN_MODES = {'AND', 'OR', 'NOT'}


def _parse_top_k(value):
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        raise ValueError('k must be an integer')

    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f'k must be between 1 and {MAX_TOP_K}')
    return top_k

@app.route('/')
def index():
    return render_template('index.html', stats=get_stats())

@app.route('/api/search')
def search():
    query  = request.args.get('q', '').strip()
    method = request.args.get('method', 'tfidf').lower()
    mode   = request.args.get('mode', 'AND').upper()

    if not query:
        return jsonify({'error': 'Missing query'}), 400

    if method not in VALID_METHODS:
        return jsonify({'error': f'Invalid method: {method}'}), 400

    if mode not in VALID_BOOLEAN_MODES:
        return jsonify({'error': f'Invalid boolean mode: {mode}'}), 400

    try:
        top_k = _parse_top_k(request.args.get('k', 10))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    dispatch = {
        'tfidf':        lambda: search_tfidf(query, top_k),
        'boolean':      lambda: search_boolean(query, mode, top_k),
        'lsa':          lambda: search_lsa(query, top_k),
        'bm25':         lambda: search_bm25(query, top_k),
        'all':          lambda: search_all(query, top_k),
    }

    result = dispatch[method]()
    result['query'] = query
    return jsonify(result)

@app.route('/api/stats')
def stats():
    return jsonify(get_stats())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, port=port, host='0.0.0.0')

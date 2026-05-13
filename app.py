# ============================================================
# app.py — Flask Web Application
# ============================================================
from flask import Flask, render_template, request, jsonify
from search import (
    search_tfidf, search_boolean,
    search_lsa, search_bm25,
    search_all, get_stats
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', stats=get_stats())

@app.route('/api/search')
def search():
    query  = request.args.get('q', '').strip()
    method = request.args.get('method', 'tfidf')
    top_k  = int(request.args.get('k', 10))
    mode   = request.args.get('mode', 'AND')

    if not query:
        return jsonify({'error': 'Missing query'}), 400

    dispatch = {
        'tfidf':        lambda: search_tfidf(query, top_k),
        'boolean':      lambda: search_boolean(query, mode, top_k),
        'lsa':          lambda: search_lsa(query, top_k),
        'bm25':         lambda: search_bm25(query, top_k),
        'all':          lambda: search_all(query, top_k),
    }

    if method not in dispatch:
        return jsonify({'error': f'Invalid method: {method}'}), 400

    result = dispatch[method]()
    result['query'] = query
    return jsonify(result)

@app.route('/api/stats')
def stats():
    return jsonify(get_stats())

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')

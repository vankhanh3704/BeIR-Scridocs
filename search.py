# ============================================================
# search.py — Load models và hàm tìm kiếm cho cả 3 thành viên
# ============================================================
import os, re, pickle, time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── PATHS — khớp với cấu trúc thư mục của 3 thành viên ─────
PROCESSED_DIR = "data/processed"
RAW_DIR       = "data/raw"
MODELS_DIR    = "models"

# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════
print("Loading data...")

# Corpus TV1
df = pd.read_parquet(f"{PROCESSED_DIR}/data_tv1.parquet")
df = df.dropna(subset=['text_tfidf']).reset_index(drop=True)

# Queries và Qrels
queries_df   = pd.read_parquet(f"{RAW_DIR}/queries.parquet")
qrels_df     = pd.read_parquet(f"{RAW_DIR}/qrels.parquet")
queries_dict = dict(zip(queries_df['_id'].astype(str), queries_df['text']))

print(f"  Corpus: {len(df):,} papers")

# ════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)

PS   = PorterStemmer()
STOP = set(stopwords.words('english'))
CUSTOM_STOP = {
    'paper','propos','method','approach','result',
    'show','base','also','howev','therefor','thu',
    'furthermor','present','work','studi',
    'experiment','evalu','use','new','high',
}
PATTERN_NON_ALPHA = re.compile(r'[^a-z\s]')

def preprocess_query(query: str) -> str:
    query = str(query).lower()
    query = PATTERN_NON_ALPHA.sub(' ', query)
    tokens = [t for t in query.split() if t not in STOP and len(t) > 2]
    stemmed = [PS.stem(t) for t in tokens]
    return ' '.join(t for t in stemmed if t not in CUSTOM_STOP)

# ════════════════════════════════════════════════════════════
# TV1: LOAD TF-IDF + BUILD INVERTED INDEX
# ════════════════════════════════════════════════════════════
print("Loading TV1 (TF-IDF)...")
from sklearn.feature_extraction.text import TfidfVectorizer

TV1_VEC_PATH  = f"{MODELS_DIR}/tv1_vectorizer.pkl"
TV1_MAT_PATH  = f"{MODELS_DIR}/tv1_matrix.pkl"

if os.path.exists(TV1_VEC_PATH):
    tfidf_vec = pickle.load(open(TV1_VEC_PATH, 'rb'))
    tfidf_mat = pickle.load(open(TV1_MAT_PATH, 'rb'))
else:
    print("  Building TF-IDF matrix...")
    tfidf_vec = TfidfVectorizer(
        max_features=30000, ngram_range=(1,2),
        min_df=2, max_df=0.85, sublinear_tf=True
    )
    tfidf_mat = tfidf_vec.fit_transform(df['text_tfidf'])
    pickle.dump(tfidf_vec, open(TV1_VEC_PATH, 'wb'))
    pickle.dump(tfidf_mat, open(TV1_MAT_PATH, 'wb'))

# Inverted index cho Boolean
print("  Building inverted index...")
inverted_index = {}
for idx, text in enumerate(df['text_tfidf']):
    if pd.isna(text): continue
    for word in set(str(text).split()):
        if word not in inverted_index:
            inverted_index[word] = set()
        inverted_index[word].add(idx)

doc_ids_tv1 = df['id'].astype(str).tolist()
print(f"  TV1 ready: {tfidf_mat.shape}")

# ════════════════════════════════════════════════════════════
# TV2: LOAD LSA MODEL
# ════════════════════════════════════════════════════════════
print("Loading TV2 (LSA)...")
try:
    lsa_model = pickle.load(open(f"{MODELS_DIR}/lsa_model.pkl", 'rb'))
    lsa_vectorizer = lsa_model['vectorizer']
    lsa_svd        = lsa_model['svd']
    lsa_matrix     = lsa_model['lsa_matrix']
    doc_ids_lsa    = lsa_model['doc_ids']
    # Nếu bạn có lưu abstract cho TV2 thì load ở đây
    print(f"  LSA ready: {lsa_matrix.shape}")
except:
    print("  LSA model not found — TV2 disabled")
    # BẠN ĐANG THIẾU DÒNG NÀY: Phải khai báo None cho các biến nếu không có file
    lsa_vectorizer = None
    lsa_svd        = None
    lsa_matrix     = None   # <--- THÊM DÒNG NÀY VÀO
    doc_ids_lsa    = []
# ════════════════════════════════════════════════════════════
# TV3: LOAD BM25
# ════════════════════════════════════════════════════════════
print("Loading TV3 (BM25)...")
BM25_MODEL_PATH  = f"{MODELS_DIR}/bm25_model.pkl"
BM25_DOCIDS_PATH = f"{MODELS_DIR}/bm25_doc_ids.csv"

bm25_model  = None
doc_ids_bm25= None
if os.path.exists(BM25_MODEL_PATH):
    bm25_model   = pickle.load(open(BM25_MODEL_PATH, 'rb'))
    bm25_docids_df = pd.read_csv(BM25_DOCIDS_PATH)
    doc_ids_bm25 = bm25_docids_df['doc_id'].astype(str).tolist()
    print(f"  BM25 ready: {len(doc_ids_bm25):,} docs")
else:
    print("  BM25 model not found — TV3 disabled")

print("All models loaded!")

# ════════════════════════════════════════════════════════════
# FORMAT HELPER
# ════════════════════════════════════════════════════════════
def _fmt(df_slice, scores, method) -> list:
    results = []
    for i, (_, row) in enumerate(df_slice.iterrows()):
        results.append({
            'rank':     i + 1,
            'id':       str(row.get('id', row.get('_id', ''))),
            'title':    str(row.get('title', '')),
            'abstract': str(row.get('abstract', ''))[:250] + '...',
            # SỬA DÒNG DƯỚI ĐÂY: Thêm "is not None" vào để tránh lỗi Numpy Array
            'score':    round(float(scores[i]), 4) if scores is not None else None,
            'method':   method,
        })
    return results

# ════════════════════════════════════════════════════════════
# TV1: SEARCH FUNCTIONS
# ════════════════════════════════════════════════════════════
def search_tfidf(query: str, top_k=10) -> dict:
    t0    = time.time()
    q_proc = preprocess_query(query)
    if not q_proc:
        return {'results': [], 'method': 'TF-IDF', 'time_ms': 0}

    q_vec  = tfidf_vec.transform([q_proc])
    scores = cosine_similarity(q_vec, tfidf_mat).flatten()
    nonzero = scores.nonzero()[0]
    if len(nonzero) == 0:
        return {'results': [], 'method': 'TF-IDF', 'time_ms': 0}

    top_idx = nonzero[scores[nonzero].argsort()[::-1][:top_k]]
    res     = df.iloc[top_idx][['id','title','abstract']].copy()
    sc      = scores[top_idx]

    return {
        'results':  _fmt(res, sc, 'TF-IDF'),
        'method':   'TF-IDF',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

def search_boolean(query: str, mode='AND', top_k=10) -> dict:
    t0     = time.time()
    tokens = preprocess_query(query).split()
    if not tokens:
        return {'results': [], 'method': f'Boolean-{mode}', 'time_ms': 0}

    valid_tokens = [t for t in tokens if t in inverted_index]

    if mode == 'AND':
        if len(valid_tokens) < len(tokens):
            return {'results': [], 'method': 'Boolean-AND', 'time_ms': 0}
        match_idx = list(set.intersection(
            *[inverted_index[t] for t in valid_tokens]
        ))
    elif mode == 'OR':
        match_idx = list(set.union(
            *[inverted_index[t] for t in valid_tokens]
        )) if valid_tokens else []
    elif mode == 'NOT':
        if len(tokens) >= 2:
            have = inverted_index.get(tokens[0], set())
            not_ = inverted_index.get(tokens[1], set())
            match_idx = list(have - not_)
        else:
            match_idx = list(inverted_index.get(tokens[0], set()))
    else:
        match_idx = []

    if not match_idx:
        return {'results': [], 'method': f'Boolean-{mode}', 'time_ms': 0}

    res = df.iloc[match_idx][['id','title','abstract']].copy()
    res['_ms'] = res.index.map(
        lambda i: sum(1 for t in valid_tokens if i in inverted_index[t])
    )
    res = res.sort_values('_ms', ascending=False).head(top_k)
    sc  = res['_ms'].tolist()
    res = res.drop(columns=['_ms'])

    return {
        'results':  _fmt(res, sc, f'Boolean-{mode}'),
        'method':   f'Boolean-{mode}',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

# ════════════════════════════════════════════════════════════
# TV2: LSA SEARCH
# ════════════════════════════════════════════════════════════
def search_lsa(query: str, top_k=10) -> dict:
    if lsa_matrix is None:
        return {'results': [], 'method': 'LSA', 'time_ms': 0,
                'error': 'LSA model not loaded'}
    t0    = time.time()
    q_vec = lsa_svd.transform(lsa_vectorizer.transform([query]))
    q_vec = normalize(q_vec, axis=1)
    scores= np.dot(q_vec, lsa_matrix.T).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        doc_id = doc_ids_lsa[idx]
        row = df[df['id'].astype(str) == doc_id]
        title    = str(row['title'].values[0])    if len(row) > 0 else ''
        abstract = str(row['abstract'].values[0]) if len(row) > 0 else ''
        results.append({
            'rank':     rank + 1,
            'id':       doc_id,
            'title':    str(title),
            'abstract': str(abstract)[:250] + '...',
            'score':    round(float(scores[idx]), 4),
            'method':   'LSA',
        })

    return {
        'results':  results,
        'method':   'LSA',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

# ════════════════════════════════════════════════════════════
# TV3: BM25 SEARCH
# ════════════════════════════════════════════════════════════
def search_bm25(query: str, top_k=10) -> dict:
    if bm25_model is None:
        return {'results': [], 'method': 'BM25', 'time_ms': 0,
                'error': 'BM25 model not loaded'}
    t0     = time.time()
    tokens = preprocess_query(query).split()
    if not tokens:
        return {'results': [], 'method': 'BM25', 'time_ms': 0}

    scores  = np.array(bm25_model.get_scores(tokens))
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        doc_id = doc_ids_bm25[idx]
        row = df[df['id'].astype(str) == doc_id]
        title    = row['title'].values[0]    if len(row) else ''
        abstract = row['abstract'].values[0] if len(row) else ''
        results.append({
            'rank':     rank + 1,
            'id':       doc_id,
            'title':    str(title),
            'abstract': str(abstract)[:250] + '...',
            'score':    round(float(scores[idx]), 4),
            'method':   'BM25',
        })

    return {
        'results':  results,
        'method':   'BM25',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

# ════════════════════════════════════════════════════════════
# COMPARE ALL
# ════════════════════════════════════════════════════════════
def search_all(query: str, top_k=5) -> dict:
    return {
        'tfidf':   search_tfidf(query,   top_k),
        'boolean': search_boolean(query, 'AND', top_k),
        'lsa':     search_lsa(query,     top_k),
        'bm25':    search_bm25(query,    top_k),
    }

def get_stats() -> dict:
    return {
        'corpus_size': len(df),
        'queries':     len(queries_dict),
        'tv1_ready':   tfidf_mat is not None,
        'tv2_ready':   lsa_matrix is not None,
        'tv3_ready':   bm25_model is not None,
    }

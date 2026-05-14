# ============================================================
# search.py - Load models and expose search functions
# ============================================================
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"

TV1_VEC_PATH = MODELS_DIR / "tv1_vectorizer.pkl"
TV1_MAT_PATH = MODELS_DIR / "tv1_matrix.pkl"
LSA_MODEL_PATH = MODELS_DIR / "lsa_model.pkl"
BM25_MODEL_PATH = MODELS_DIR / "bm25_model.pkl"
BM25_DOCIDS_PATH = MODELS_DIR / "bm25_doc_ids.csv"


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _save_pickle(obj, path: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def _load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set(ENGLISH_STOP_WORDS)


def _clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _shorten(value, limit: int = 250) -> str:
    text = _clean_text(value)
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")

df = pd.read_parquet(PROCESSED_DIR / "data_tv1.parquet")
df = df.dropna(subset=["text_tfidf"]).reset_index(drop=True)

queries_df = pd.read_parquet(RAW_DIR / "queries.parquet")
qrels_df = pd.read_parquet(RAW_DIR / "qrels.parquet")
queries_dict = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))

doc_lookup = (
    df.assign(_id_str=df["id"].astype(str))
    .set_index("_id_str")[["title", "abstract"]]
    .to_dict("index")
)

print(f"  Corpus: {len(df):,} papers")


# ============================================================
# PREPROCESSING
# ============================================================
PS = PorterStemmer()
STOP = _load_stopwords()
CUSTOM_STOP = {
    "paper", "propos", "method", "approach", "result",
    "show", "base", "also", "howev", "therefor", "thu",
    "furthermor", "present", "work", "studi",
    "experiment", "evalu", "use", "new", "high",
}
CUSTOM_STOP_BM25 = {
    "paper", "propos", "method", "approach", "result",
    "show", "base", "also", "howev", "therefor", "thu",
    "furthermor", "present", "work", "studi",
    "experiment", "evalu",
}
PATTERN_NON_ALPHA = re.compile(r"[^a-z\s]")
PATTERN_BM25_CLEAN = re.compile(r"[^a-z0-9\s]")
PATTERN_BM25_SPACE = re.compile(r"\s+")
PATTERN_LSA_QUERY = re.compile(r"[^a-zA-Z0-9\s\-\.\,]")


def preprocess_query(query: str) -> str:
    query = str(query).lower()
    query = PATTERN_NON_ALPHA.sub(" ", query)
    tokens = [t for t in query.split() if t not in STOP and len(t) > 1]
    stemmed = [PS.stem(t) for t in tokens]
    return " ".join(t for t in stemmed if t not in CUSTOM_STOP)


def preprocess_bm25_query(query: str) -> list[str]:
    text = str(query).lower()
    text = PATTERN_BM25_CLEAN.sub(" ", text)
    text = PATTERN_BM25_SPACE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t not in STOP and len(t) > 1]
    stemmed = [PS.stem(t) for t in tokens]
    return [t for t in stemmed if t not in CUSTOM_STOP_BM25]


def preprocess_lsa_query(query: str) -> str:
    query = PATTERN_LSA_QUERY.sub(" ", str(query).lower())
    tokens = [t for t in query.split() if len(t) > 1 and t not in STOP]
    return " ".join(tokens)


# ============================================================
# TV1: TF-IDF + Boolean
# ============================================================
print("Loading TV1 (TF-IDF)...")

if TV1_VEC_PATH.exists() and TV1_MAT_PATH.exists():
    tfidf_vec = _load_pickle(TV1_VEC_PATH)
    tfidf_mat = _load_pickle(TV1_MAT_PATH)
else:
    print("  Building TF-IDF matrix...")
    tfidf_vec = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )
    tfidf_mat = tfidf_vec.fit_transform(df["text_tfidf"])
    _save_pickle(tfidf_vec, TV1_VEC_PATH)
    _save_pickle(tfidf_mat, TV1_MAT_PATH)

print("  Building inverted index...")
inverted_index: dict[str, set[int]] = {}
for idx, text in enumerate(df["text_tfidf"]):
    for word in set(str(text).split()):
        inverted_index.setdefault(word, set()).add(idx)

doc_ids_tv1 = df["id"].astype(str).tolist()
print(f"  TV1 ready: {tfidf_mat.shape}")


# ============================================================
# TV2: LSA
# ============================================================
print("Loading TV2 (LSA)...")
lsa_vectorizer = None
lsa_svd = None
lsa_matrix = None
doc_ids_lsa: list[str] = []

try:
    if not LSA_MODEL_PATH.exists():
        raise FileNotFoundError(LSA_MODEL_PATH.name)
    lsa_model = _load_pickle(LSA_MODEL_PATH)
    lsa_vectorizer = lsa_model["vectorizer"]
    lsa_svd = lsa_model["svd"]
    lsa_matrix = np.asarray(lsa_model["lsa_matrix"])
    doc_ids_lsa = [str(doc_id) for doc_id in lsa_model["doc_ids"]]
    print(f"  LSA ready: {lsa_matrix.shape}")
except Exception as exc:
    print(f"  LSA disabled: {exc}")


# ============================================================
# TV3: BM25
# ============================================================
print("Loading TV3 (BM25)...")
bm25_model = None
doc_ids_bm25: list[str] = []

try:
    if not BM25_MODEL_PATH.exists() or not BM25_DOCIDS_PATH.exists():
        raise FileNotFoundError("bm25_model.pkl or bm25_doc_ids.csv")
    bm25_model = _load_pickle(BM25_MODEL_PATH)
    bm25_docids_df = pd.read_csv(BM25_DOCIDS_PATH)
    doc_ids_bm25 = bm25_docids_df["doc_id"].astype(str).tolist()
    print(f"  BM25 ready: {len(doc_ids_bm25):,} docs")
except Exception as exc:
    print(f"  BM25 disabled: {exc}")

print("All models loaded!")


# ============================================================
# HELPERS
# ============================================================
def _doc_by_id(doc_id: str) -> dict[str, str]:
    row = doc_lookup.get(str(doc_id), {})
    return {
        "title": _clean_text(row.get("title", "")),
        "abstract": _shorten(row.get("abstract", "")),
    }


def _fmt(df_slice: pd.DataFrame, scores, method: str) -> list[dict]:
    results = []
    for rank, (_, row) in enumerate(df_slice.iterrows(), start=1):
        score = scores[rank - 1] if scores is not None else None
        results.append({
            "rank": rank,
            "id": str(row.get("id", row.get("_id", ""))),
            "title": _clean_text(row.get("title", "")),
            "abstract": _shorten(row.get("abstract", "")),
            "score": round(float(score), 4) if score is not None else None,
            "method": method,
        })
    return results


from typing import Union

def _empty(method: str, t0: Union[float, None] = None, error: Union[str, None] = None) -> dict:
    payload = {
        "results": [],
        "method": method,
        "time_ms": 0 if t0 is None else round((time.time() - t0) * 1000, 1),
    }
    if error:
        payload["error"] = error
    return payload


def _load_eval_summary() -> dict:
    summary: dict[str, dict[str, float]] = {}

    try:
        path = MODELS_DIR / "tv1_evaluation.csv"
        if path.exists():
            eval_df = pd.read_csv(path)
            tfidf_df = eval_df[eval_df["method"].astype(str).str.upper() == "TFIDF"]
            row10 = tfidf_df[tfidf_df["k"] == 10].mean(numeric_only=True)
            summary["tfidf"] = {
                "p10": round(float(row10.get("P@K", 0)), 4),
                "r10": round(float(row10.get("R@K", 0)), 4),
                "ndcg10": round(float(row10.get("NDCG@K", 0)), 4),
                "map": round(float(tfidf_df["AP"].mean()), 4),
            }
    except Exception as exc:
        summary["tfidf"] = {"error": str(exc)}

    try:
        path = MODELS_DIR / "bm25_eval_metrics.csv"
        if path.exists():
            eval_df = pd.read_csv(path)
            row10 = eval_df[eval_df["Top_K"] == 10].iloc[0]
            summary["bm25"] = {
                "p10": round(float(row10["Precision"]), 4),
                "r10": round(float(row10["Recall"]), 4),
                "ndcg10": round(float(row10["NDCG"]), 4),
            }
    except Exception as exc:
        summary["bm25"] = {"error": str(exc)}

    return summary


# ============================================================
# TV1 SEARCH
# ============================================================
def search_tfidf(query: str, top_k=10) -> dict:
    t0 = time.time()
    q_proc = preprocess_query(query)
    if not q_proc:
        return _empty("TF-IDF", t0)

    q_vec = tfidf_vec.transform([q_proc])
    scores = cosine_similarity(q_vec, tfidf_mat).flatten()
    nonzero = scores.nonzero()[0]
    if len(nonzero) == 0:
        return _empty("TF-IDF", t0)

    top_idx = nonzero[scores[nonzero].argsort()[::-1][:top_k]]
    res = df.iloc[top_idx][["id", "title", "abstract"]].copy()

    return {
        "results": _fmt(res, scores[top_idx], "TF-IDF"),
        "method": "TF-IDF",
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


def search_boolean(query: str, mode="AND", top_k=10) -> dict:
    t0 = time.time()
    mode = str(mode or "AND").upper()
    tokens = preprocess_query(query).split()
    if not tokens:
        return _empty(f"Boolean-{mode}", t0)

    valid_tokens = [t for t in tokens if t in inverted_index]
    ranking_tokens = valid_tokens

    if mode == "AND":
        if len(valid_tokens) < len(tokens):
            return _empty("Boolean-AND", t0)
        match_idx = set.intersection(*[inverted_index[t] for t in valid_tokens])
    elif mode == "OR":
        match_idx = set.union(*[inverted_index[t] for t in valid_tokens]) if valid_tokens else set()
    elif mode == "NOT":
        include = tokens[0]
        if include not in inverted_index:
            return _empty("Boolean-NOT", t0)
        match_idx = set(inverted_index[include])
        for excluded in tokens[1:]:
            match_idx -= inverted_index.get(excluded, set())
        ranking_tokens = [include]
    else:
        return _empty(f"Boolean-{mode}", t0, error=f"Invalid boolean mode: {mode}")

    if not match_idx:
        return _empty(f"Boolean-{mode}", t0)

    match_idx = list(match_idx)
    rank_scores = np.zeros(len(match_idx))
    if ranking_tokens:
        q_vec = tfidf_vec.transform([" ".join(ranking_tokens)])
        rank_scores = cosine_similarity(q_vec, tfidf_mat[match_idx]).flatten()

    res = df.iloc[match_idx][["id", "title", "abstract"]].copy()
    res["_rank_score"] = rank_scores
    res["_matches"] = res.index.map(
        lambda i: sum(1 for token in ranking_tokens if i in inverted_index.get(token, set()))
    )
    res = res.sort_values(
        ["_rank_score", "_matches", "id"],
        ascending=[False, False, True],
    ).head(top_k)
    scores = res["_matches"].tolist()
    res = res.drop(columns=["_rank_score", "_matches"])

    return {
        "results": _fmt(res, scores, f"Boolean-{mode}"),
        "method": f"Boolean-{mode}",
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


# ============================================================
# TV2 SEARCH
# ============================================================
def search_lsa(query: str, top_k=10) -> dict:
    if lsa_matrix is None:
        return _empty("LSA", error="LSA model not loaded")

    t0 = time.time()
    q_proc = preprocess_lsa_query(query)
    if not q_proc:
        return _empty("LSA", t0)

    q_vec = lsa_svd.transform(lsa_vectorizer.transform([q_proc]))
    q_vec = normalize(q_vec, axis=1)
    if not np.any(q_vec):
        return _empty("LSA", t0)

    scores = np.dot(q_vec, lsa_matrix.T).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        doc_id = doc_ids_lsa[idx]
        doc = _doc_by_id(doc_id)
        results.append({
            "rank": rank,
            "id": doc_id,
            "title": doc["title"],
            "abstract": doc["abstract"],
            "score": round(float(scores[idx]), 4),
            "method": "LSA",
        })

    return {
        "results": results,
        "method": "LSA",
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


# ============================================================
# TV3 SEARCH
# ============================================================
def search_bm25(query: str, top_k=10) -> dict:
    if bm25_model is None:
        return _empty("BM25", error="BM25 model not loaded")

    t0 = time.time()
    tokens = preprocess_bm25_query(query)
    if not tokens:
        return _empty("BM25", t0)

    scores = np.asarray(bm25_model.get_scores(tokens), dtype=float)
    nonzero = scores.nonzero()[0]
    if len(nonzero) == 0:
        return _empty("BM25", t0)

    top_idx = nonzero[scores[nonzero].argsort()[::-1][:top_k]]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        doc_id = doc_ids_bm25[idx]
        doc = _doc_by_id(doc_id)
        results.append({
            "rank": rank,
            "id": doc_id,
            "title": doc["title"],
            "abstract": doc["abstract"],
            "score": round(float(scores[idx]), 4),
            "method": "BM25",
        })

    return {
        "results": results,
        "method": "BM25",
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


# ============================================================
# COMPARE + STATS
# ============================================================
def search_all(query: str, top_k=5) -> dict:
    t0 = time.time()
    result = {
        "tfidf": search_tfidf(query, top_k),
        "boolean": search_boolean(query, "AND", top_k),
        "lsa": search_lsa(query, top_k),
        "bm25": search_bm25(query, top_k),
        "method": "ALL",
    }
    result["time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


def get_stats() -> dict:
    return {
        "corpus_size": len(df),
        "queries": len(queries_dict),
        "qrels": len(qrels_df),
        "tv1_ready": tfidf_mat is not None,
        "tv2_ready": lsa_matrix is not None,
        "tv3_ready": bm25_model is not None,
        "evaluation": _load_eval_summary(),
    }

# SciFind — SCIDOCS IR Web App

## Cấu trúc thư mục

```
scidocs_webapp/
├── app.py              ← Flask routes
├── search.py           ← Load models + search logic
├── requirements.txt
├── templates/
│   └── index.html      ← Giao diện web
├── data/
│   ├── raw/
│   │   ├── queries.parquet
│   │   └── qrels.parquet
│   └── processed/
│       ├── data_tv1.parquet    ← từ preprocessed notebook
│       ├── data_tv2.parquet
│       └── corpus_bm25.parquet
└── models/
    ├── tv1_vectorizer.pkl      ← tự build lần đầu
    ├── tv1_matrix.pkl          ← tự build lần đầu
    ├── lsa_model.pkl           ← từ TV2 notebook
    ├── bm25_model.pkl          ← từ TV3 notebook
    └── bm25_doc_ids.csv        ← từ TV3 notebook
```

## Setup

```bash
pip install -r requirements.txt
python app.py
# Truy cập: http://localhost:5000
```

## Lấy model files

Từ notebook TV2 (lsa.ipynb):
```python
# Lưu vào models/lsa_model.pkl
model_data = {
    'vectorizer': lsa_engine.vectorizer,
    'svd':        lsa_engine.svd,
    'lsa_matrix': lsa_engine.lsa_matrix,
    'doc_ids':    lsa_engine.doc_ids,
    'titles':     df_lsa['title'].tolist(),
}
pickle.dump(model_data, open('models/lsa_model.pkl', 'wb'))
```

Từ notebook TV3 (bm25.ipynb):
```python
# Đã lưu sẵn: models/bm25_model.pkl và models/bm25_doc_ids.csv
```

## Ghi chú

- TF-IDF vectorizer và matrix tự build lần đầu chạy app, lưu vào models/
- Lần sau load từ disk, không cần build lại
- LSA cần copy models/lsa_model.pkl từ TV2 notebook
- BM25 cần copy models/bm25_model.pkl và bm25_doc_ids.csv từ TV3 notebook

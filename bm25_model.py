import math
import numpy as np
from collections import Counter

class CustomBM25:

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.inverted_index = {}
        self.idf = {}
                
        # Buoc 1: Xay dung Inverted Index va tinh do dai tai lieu
        total_len = 0
        for doc_idx, doc in enumerate(corpus):
            doc_len = len(doc)
            self.doc_lengths.append(doc_len)
            total_len += doc_len
            
            # Đếm tần suất từ khóa trong tai lieu hien tai
            term_frequencies = Counter(doc)
            for term, freq in term_frequencies.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_idx] = freq
                
        self.avgdl = total_len / self.corpus_size
        
        # Buoc 2: Tinh toan do hiem cua tu (Inverse Document Frequency - IDF)
        for term, docs_containing_term in self.inverted_index.items():
            n_q = len(docs_containing_term)
            # Cong thuc IDF tieu chuan cua BM25 Okapi
            idf_value = math.log(((self.corpus_size - n_q + 0.5) / (n_q + 0.5)) + 1.0)
            self.idf[term] = idf_value
        

    def get_scores(self, query):
        
        scores = np.zeros(self.corpus_size)
        
        for term in query:
            # Bo qua cac tu khong ton tai trong tap tu vung
            if term not in self.inverted_index:
                continue
                
            q_idf = self.idf[term]
            
            # Chi duyet qua cac tai lieu co chua tu khoa nay
            for doc_idx, freq in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_idx]
                
                # Cong thuc tinh diem BM25
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                
                scores[doc_idx] += q_idf * (numerator / denominator)
                
        return scores
from sklearn.feature_extraction.text import TfidfVectorizer
from .scoring import occ

_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(occ["title"])
_M = _vec.transform(occ["title"])

def top_matches(text: str, k: int = 3):
    sims = (_M @ _vec.transform([text]).T).toarray().ravel()
    idx = sims.argsort()[::-1][:k]
    return [(occ.index[i], occ["title"].iloc[i], float(sims[i])) for i in idx]
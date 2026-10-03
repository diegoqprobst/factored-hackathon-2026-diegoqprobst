"""Probabilistic router: one feature extractor per head, logistic-regression heads for intent, injection and
language. Features are either TF-IDF (char + word n-grams) or multilingual-e5-small sentence embeddings."""
import json
import threading
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from src.router.keyword import KeywordRouter
from src.router.labels import MAX_CHARS, Example, RouterResult, normalize


E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # pinned HF commit of intfloat/multilingual-e5-small


def tfidf_features() -> FeatureUnion:
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, preprocessor=normalize)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, preprocessor=normalize)),
    ])


_ENCODER_LOCK = threading.Lock()


def _load_encoder(model_name: str, revision: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, revision=revision)
    return lambda ts: model.encode(ts, normalize_embeddings=True, batch_size=64)


class E5Featurizer(BaseEstimator, TransformerMixin):
    """Sentence embeddings at a pinned model revision; the encoder is loaded once (thread-safe), lazily, and is
    never pickled (the artifact stays small)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", revision: str = E5_REVISION,
                 encoder=None):
        self.model_name = model_name
        self.revision = revision
        self.encoder = encoder

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.encoder is None:
            with _ENCODER_LOCK:
                if self.encoder is None:
                    self.encoder = _load_encoder(self.model_name, getattr(self, "revision", E5_REVISION))
        return np.asarray(self.encoder(["query: " + t for t in X]))

    def __getstate__(self):
        state = super().__getstate__()
        state["encoder"] = None
        return state


def _lr() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")


class ClassifierRouter:
    def __init__(self, make_features, version: str):
        self.version = version
        self.threshold = 0.0
        self.intent = make_pipeline(make_features(), _lr())
        self.injection = make_pipeline(make_features(), _lr())
        self.language = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), preprocessor=normalize),
            LogisticRegression(max_iter=3000))

    def fit(self, examples: list[Example]) -> "ClassifierRouter":
        texts = [e.text[:MAX_CHARS] for e in examples]
        self.intent.fit(texts, [e.intent for e in examples])
        self.injection.fit(texts, [int(e.injection) for e in examples])
        self.language.fit(texts, [e.language for e in examples])
        return self

    def predict_many(self, texts: list[str]) -> list[RouterResult]:
        texts = [(t or "")[:MAX_CHARS] for t in texts]
        p_intent = self.intent.predict_proba(texts)
        classes = self.intent.classes_
        inj_col = list(self.injection.classes_).index(1) if 1 in self.injection.classes_ else None
        p_inj = self.injection.predict_proba(texts)[:, inj_col] if inj_col is not None else np.zeros(len(texts))
        langs = self.language.predict(texts)
        out = []
        for k, text in enumerate(texts):
            j = int(np.argmax(p_intent[k]))
            confidence = float(p_intent[k, j])
            blank = not any(ch.isalnum() for ch in text)
            out.append(RouterResult(
                intent="oos_other" if blank else str(classes[j]),
                confidence=0.0 if blank else confidence,
                language=str(langs[k]),
                injection=bool(p_inj[k] >= 0.5) and not blank,
                injection_score=float(p_inj[k]),
                abstain=blank or confidence < self.threshold,
                model_version=self.version))
        return out

    def predict(self, text: str) -> RouterResult:
        return self.predict_many([text])[0]

    def save(self, path: Path, meta: dict) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "router.joblib")
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def tfidf_router() -> ClassifierRouter:
    return ClassifierRouter(tfidf_features, "tfidf_v1")


def e5_router(encoder=None) -> ClassifierRouter:
    shared = E5Featurizer(encoder=encoder)  # embeddings are not fitted, so both heads can share one encoder
    return ClassifierRouter(lambda: shared, "e5_v1")


def load_router(path: Path = Path("models/router_v1")):
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"router artifact not found at {path} (run `make router`)")
    meta = json.loads(meta_path.read_text())
    if meta["model"] == "keyword":
        return KeywordRouter()
    router = joblib.load(path / "router.joblib")  # pickle: only ever load repo-controlled artifacts
    try:
        router.predict_many(["warmup"])  # fail at startup, not on the first customer message; loads the encoder
    except ImportError as exc:
        raise RuntimeError(f"router '{meta['model']}' needs the optional 'embeddings' dependency group "
                           "(uv sync --group embeddings), or ship the tfidf router instead") from exc
    return router

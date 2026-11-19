"""TF-IDF baselines for aspect detection and aspect sentiment classification."""

from typing import Literal

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from absa.data import ASPECTS

# Words of any length, plus each emoji as its own token (same ranges as preprocess strip_emoji).
WORD_TOKEN_PATTERN = r"(?u)\b\w+\b|[\U0001F000-\U0001FAFF☀-➿⬀-⯿]"


class TfidfAbsa:
    """Train one classifier per aspect, with optional separate sentiment classifiers."""

    def __init__(
        self,
        mode: Literal["split", "joint"] = "split",
        analyzer: Literal["word", "char_wb"] = "word",
        ngram_range: tuple[int, int] | list[int] = (1, 2),
        min_df: int = 2,
        classifier: Literal["lr", "svm"] = "lr",
        C: float = 1.0,
        seed: int = 42,
    ) -> None:
        if mode not in ("split", "joint"):
            raise ValueError(f"mode không hợp lệ: {mode}")
        if classifier not in ("lr", "svm"):
            raise ValueError(f"classifier không hợp lệ: {classifier}")
        self.mode = mode
        self.classifier = classifier
        self.C = C
        self.seed = seed
        # sklearn's default token_pattern drops emoji and 1-char words ("ý"), which made E3 a no-op.
        extra = {"token_pattern": WORD_TOKEN_PATTERN} if analyzer == "word" else {}
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=min_df,
            analyzer=analyzer,
            ngram_range=tuple(ngram_range),
            **extra,
        )
        self.aspect_models: list = []
        self.sentiment_models: list = []

    def _fit_classifier(self, X, y: np.ndarray):
        if len(np.unique(y)) == 1:
            model = DummyClassifier(strategy="most_frequent")
        elif self.classifier == "lr":
            model = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        else:
            model = LinearSVC(C=self.C, random_state=self.seed)
        return model.fit(X, y)

    def fit(
        self,
        texts: list[str],
        Y: np.ndarray,
        dev_texts: list[str] | None = None,
        Y_dev: np.ndarray | None = None,
    ) -> "TfidfAbsa":
        """Fit the vectorizer and all classifiers on Train only."""
        if Y.shape != (len(texts), len(ASPECTS)):
            raise ValueError(f"Y phải có shape ({len(texts)}, {len(ASPECTS)})")
        X = self.vectorizer.fit_transform(texts)
        self.aspect_models = []
        self.sentiment_models = []
        for column in range(len(ASPECTS)):
            labels = Y[:, column]
            if self.mode == "joint":
                self.aspect_models.append(self._fit_classifier(X, labels))
                continue
            present = labels > 0
            self.aspect_models.append(self._fit_classifier(X, present))
            sentiment = self._fit_classifier(X[present], labels[present]) if present.any() else None
            self.sentiment_models.append(sentiment)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return codes 0..3 in the fixed aspect order."""
        if len(self.aspect_models) != len(ASPECTS):
            raise ValueError("Mô hình chưa được huấn luyện")
        if not texts:
            return np.zeros((0, len(ASPECTS)), dtype=np.int8)
        X = self.vectorizer.transform(texts)
        predictions = np.zeros((len(texts), len(ASPECTS)), dtype=np.int8)
        for column, aspect_model in enumerate(self.aspect_models):
            if self.mode == "joint":
                predictions[:, column] = aspect_model.predict(X)
                continue
            present = aspect_model.predict(X).astype(bool)
            sentiment = self.sentiment_models[column]
            if present.any() and sentiment is not None:
                predictions[present, column] = sentiment.predict(X[present])
        return predictions

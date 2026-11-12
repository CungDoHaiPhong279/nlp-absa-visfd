"""Chỉ số đánh giá ABSA: khía cạnh, cảm xúc, ma trận nhầm lẫn."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from absa.data import ASPECTS


def _prf(t: np.ndarray, p: np.ndarray):
    """P/R/F1/support theo từng cột nhị phân và micro-F1 (không chia 0)."""
    pr, rc, f1, sup = precision_recall_fscore_support(t, p, average=None, zero_division=0)
    micro = precision_recall_fscore_support(t, p, average="micro", zero_division=0)[2]
    return pr, rc, f1, sup, float(micro)


def _macro(f1: np.ndarray, sup: np.ndarray) -> float:
    """Trung bình F1 chỉ trên các cột có mẫu dương trong gold."""
    return float(f1[sup > 0].mean()) if (sup > 0).any() else 0.0


def evaluate(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    pr, rc, f1, sup, a_micro = _prf((Y_true > 0).astype(int), (Y_pred > 0).astype(int))

    ks = (1, 2, 3)
    st = np.hstack([Y_true == k for k in ks]).astype(int)
    sp = np.hstack([Y_pred == k for k in ks]).astype(int)
    _, _, s_f1, s_sup, s_micro = _prf(st, sp)

    both = (Y_true > 0) & (Y_pred > 0)
    if both.any():
        _, _, d_f1, d_sup = precision_recall_fscore_support(
            Y_true[both], Y_pred[both], labels=list(ks), average=None, zero_division=0
        )
        detected = _macro(d_f1, d_sup)
    else:
        detected = 0.0

    return {
        "aspect_macro_f1": _macro(f1, sup),
        "aspect_micro_f1": a_micro,
        "sentiment_macro_f1": _macro(s_f1, s_sup),
        "sentiment_micro_f1": s_micro,
        "polarity_macro_f1_detected": detected,
        "exact_match": float((Y_true == Y_pred).all(axis=1).mean()),
        "per_aspect": {
            a: {
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
            for i, a in enumerate(ASPECTS)
        },
    }


def confusion_by_aspect(Y_true: np.ndarray, Y_pred: np.ndarray, aspect: str) -> np.ndarray:
    i = ASPECTS.index(aspect)
    return confusion_matrix(np.asarray(Y_true)[:, i], np.asarray(Y_pred)[:, i], labels=[0, 1, 2, 3])

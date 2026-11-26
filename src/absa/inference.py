"""Inference for the Gradio demo (Task 12): load a saved checkpoint offline, predict, render."""

import html
import json
from pathlib import Path

import pandas as pd
import torch

from absa.baseline import TfidfAbsa
from absa.data import ASPECTS, matrix_to_labels
from absa.eda import aspect_polarity_table
from absa.preprocess import PRE_DEFAULTS, normalize, segment
from absa.transformer import AbsaHead, decode

ASPECT_VI = {
    "SCREEN": "Màn hình",
    "CAMERA": "Camera",
    "FEATURES": "Tính năng",
    "BATTERY": "Pin",
    "PERFORMANCE": "Hiệu năng",
    "STORAGE": "Bộ nhớ",
    "DESIGN": "Thiết kế",
    "PRICE": "Giá",
    "GENERAL": "Tổng quan",
    "SER&ACC": "Dịch vụ & phụ kiện",
}
COLORS = {"Positive": "#1a7f37", "Neutral": "#6e7781", "Negative": "#cf222e"}
POLARITY_VI = {"Positive": "Tích cực", "Neutral": "Trung tính", "Negative": "Tiêu cực"}
_TAG = "color:#fff;padding:6px 12px;border-radius:999px;font-weight:600;display:inline-block"


def _has_words(text: str) -> bool:
    """Emoji-only or punctuation-only input carries no aspect; skip the model for it."""
    return normalize(text, strip_emoji=True, strip_punct=True) != ""


class Predictor:
    """A saved `TransformerAbsa` checkpoint, rebuilt without network access."""

    @classmethod
    def from_dir(cls, path: Path) -> "Predictor":
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        path = Path(path)
        cfg = json.loads((path / "absa_config.json").read_text(encoding="utf-8"))
        if cfg.get("model_type") == "tfidf":
            return TfidfPredictor.from_dir(path)
        pre = {**PRE_DEFAULTS, **cfg.get("preprocess", {})}
        if pre["use_star"]:
            raise ValueError("Demo không có số sao")
        p = object.__new__(cls)
        p.pre = pre
        p.mode = cfg["mode"]
        p.max_length = cfg["max_length"]
        p.threshold = cfg.get("threshold", 0.5)
        p.device = "cuda" if torch.cuda.is_available() else "cpu"
        p.tokenizer = AutoTokenizer.from_pretrained(path)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(path))
        p.module = AbsaHead(encoder, p.mode, cfg.get("dropout", 0.1))
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        p.module.load_state_dict(state)
        p.module.to(p.device).eval()
        return p

    def _prepare(self, texts: list[str]) -> list[str]:
        out = [
            normalize(
                t,
                lowercase=self.pre["lowercase"],
                strip_emoji=self.pre["strip_emoji"],
                strip_punct=self.pre["strip_punct"],
            )
            for t in texts
        ]
        return segment(out, self.pre["segment"]) if self.pre["segment"] else out

    def predict(self, text: str) -> dict[str, str]:
        if not _has_words(text):
            return {}
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict[str, str]]:
        keep = [i for i, t in enumerate(texts) if _has_words(str(t))]
        result: list[dict[str, str]] = [{} for _ in texts]
        prepared = self._prepare([str(texts[i]) for i in keep])
        for start in range(0, len(keep), batch_size):
            chunk = prepared[start : start + batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.module(
                    enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                )
            Y = decode({k: v.float().cpu() for k, v in out.items()}, self.mode, self.threshold)
            for i, labels in zip(
                keep[start : start + batch_size], matrix_to_labels(Y), strict=True
            ):
                result[i] = labels
        return result


class TfidfPredictor:
    """Adapter exposing the same demo API for a saved TF-IDF baseline."""

    @classmethod
    def from_dir(cls, path: Path) -> "TfidfPredictor":
        path = Path(path)
        cfg = json.loads((path / "absa_config.json").read_text(encoding="utf-8"))
        p = object.__new__(cls)
        p.pre = {**PRE_DEFAULTS, **cfg.get("preprocess", {})}
        p.model = TfidfAbsa.load(path)
        return p

    def _prepare(self, texts: list[str]) -> list[str]:
        out = [
            normalize(
                t,
                lowercase=self.pre["lowercase"],
                strip_emoji=self.pre["strip_emoji"],
                strip_punct=self.pre["strip_punct"],
            )
            for t in texts
        ]
        return segment(out, self.pre["segment"]) if self.pre["segment"] else out

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict[str, str]]:
        del batch_size  # sklearn predicts the whole sparse matrix efficiently here
        keep = [i for i, t in enumerate(texts) if _has_words(str(t))]
        result: list[dict[str, str]] = [{} for _ in texts]
        if keep:
            Y = self.model.predict(self._prepare([str(texts[i]) for i in keep]))
            for i, labels in zip(keep, matrix_to_labels(Y), strict=True):
                result[i] = labels
        return result

    def predict(self, text: str) -> dict[str, str]:
        if not _has_words(text):
            return {}
        return self.predict_batch([text])[0]


def render_html(text: str, pred: dict[str, str]) -> str:
    parts = ['<div class="absa">']
    if text.strip():
        parts.append(f'<div class="quote">{html.escape(text)}</div>')
    if not pred:
        parts.append('<div class="res-empty slim">Không phát hiện khía cạnh nào</div></div>')
        return "".join(parts)
    counts = {p: sum(v == p for v in pred.values()) for p in COLORS}
    order = ("Negative", "Neutral", "Positive")
    strip = "".join(
        f'<span style="flex:{counts[p]};background:{COLORS[p]}"></span>' for p in order if counts[p]
    )
    words = " · ".join(
        f'<b style="color:{COLORS[p]}">{counts[p]} {POLARITY_VI[p].lower()}</b>'
        for p in order
        if counts[p]
    )
    parts.append(
        f'<div class="verdict"><div class="vt">{len(pred)} khía cạnh được nhắc: {words}</div>'
        f'<div class="strip">{strip}</div></div><div class="tags">'
    )
    for a in ASPECTS:
        if a in pred:
            p = pred[a]
            parts.append(
                f'<span class="tag" style="background:{COLORS[p]};{_TAG}">'
                f"{ASPECT_VI[a]}: {POLARITY_VI[p]}</span>"
            )
    parts.append("</div>")
    rest = [ASPECT_VI[a] for a in ASPECTS if a not in pred]
    if rest:
        parts.append(f'<div class="rest">Không nhắc tới: {" · ".join(rest)}</div>')
    parts.append("</div>")
    return "".join(parts)


def summarize_predictions(preds: list[dict[str, str]]) -> pd.DataFrame:
    return aspect_polarity_table(pd.DataFrame({"aspects": preds}))

"""Transformer ABSA: output heads, losses, decoding, class weights, threshold search,
and the `TransformerAbsa` model used by the experiment runner."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

from absa.data import ASPECTS
from absa.evaluate import evaluate

N_ASPECTS = len(ASPECTS)
MODES = ("joint", "split")
THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode không hợp lệ: {mode}. Hợp lệ: {', '.join(MODES)}")


class AbsaHead(nn.Module):
    """Encoder + linear head on the first token (<s>) of the last hidden state."""

    def __init__(
        self,
        encoder: PreTrainedModel,
        mode: Literal["joint", "split"],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _check_mode(mode)
        hidden = encoder.config.hidden_size
        self.encoder = encoder
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        if mode == "joint":
            self.classifier = nn.Linear(hidden, N_ASPECTS * 4)
        else:
            self.aspect = nn.Linear(hidden, N_ASPECTS)
            self.polarity = nn.Linear(hidden, N_ASPECTS * 3)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(states[:, 0])
        if self.mode == "joint":
            return {"logits": self.classifier(x).view(-1, N_ASPECTS, 4)}
        return {
            "aspect_logits": self.aspect(x),
            "polarity_logits": self.polarity(x).view(-1, N_ASPECTS, 3),
        }


def absa_loss(
    outputs: dict[str, Tensor],
    Y: Tensor,
    mode: str,
    class_weights: Tensor | None = None,
) -> Tensor:
    _check_mode(mode)
    if mode == "joint":
        logits = outputs["logits"]
        losses = [
            F.cross_entropy(
                logits[:, a],
                Y[:, a],
                weight=None if class_weights is None else class_weights[a].to(logits.device),
            )
            for a in range(N_ASPECTS)
        ]
        return torch.stack(losses).mean()

    if class_weights is not None:
        raise ValueError("class_weights chỉ hỗ trợ mode joint")
    present = Y > 0
    loss = F.binary_cross_entropy_with_logits(outputs["aspect_logits"], present.float())
    if present.any():
        # polarity is only learned where the aspect is really present (labels 1..3 -> 0..2)
        loss = loss + F.cross_entropy(outputs["polarity_logits"][present], Y[present] - 1)
    return loss


def decode(outputs: dict[str, Tensor], mode: str, threshold: float = 0.5) -> np.ndarray:
    _check_mode(mode)
    with torch.no_grad():
        if mode == "joint":
            codes = outputs["logits"].argmax(-1)
        else:
            present = torch.sigmoid(outputs["aspect_logits"]) > threshold
            codes = torch.where(present, outputs["polarity_logits"].argmax(-1) + 1, 0)
    return codes.cpu().numpy().astype(np.int8)


def compute_class_weights(Y: np.ndarray) -> Tensor:
    """Inverse-frequency weights per (aspect, class), add-one smoothed, row mean 1, clipped."""
    Y = np.asarray(Y)
    counts = np.stack([(Y == k).sum(axis=0) for k in range(4)], axis=1) + 1.0
    weights = 1.0 / counts
    weights = weights / weights.mean(axis=1, keepdims=True)
    return torch.tensor(np.clip(weights, 0.1, 10.0), dtype=torch.float32)


def best_threshold(
    aspect_probs: np.ndarray,
    polarity_codes: np.ndarray,
    Y_true: np.ndarray,
    grid: tuple[float, ...] = THRESHOLD_GRID,
) -> float:
    """Threshold with the best Dev sentiment macro-F1; ties go to the one closest to 0.5."""
    scored = [
        (
            evaluate(Y_true, np.where(aspect_probs > t, polarity_codes, 0))["sentiment_macro_f1"],
            -abs(t - 0.5),
            t,
        )
        for t in grid
    ]
    return max(scored)[2]


class TransformerAbsa:
    """Fine-tunes a pretrained encoder (PhoBERT, ViSoBERT) or the offline `__tiny__` model."""

    def __init__(
        self,
        model_name: str,
        mode: Literal["joint", "split"] = "joint",
        max_length: int = 256,
        lr: float = 2e-5,
        batch_size: int = 16,
        eval_batch_size: int = 64,
        epochs: int = 5,
        class_weight: bool = False,
        fp16: bool = True,
        patience: int = 2,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        _check_mode(mode)
        if class_weight and mode == "split":
            raise ValueError("class_weight chỉ hỗ trợ mode joint")
        if max_length > 256:
            raise ValueError("max_length tối đa là 256")
        self.model_name = model_name
        self.mode = mode
        self.max_length = max_length
        self.lr = lr
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.class_weight = class_weight
        self.fp16 = fp16
        self.patience = patience
        self.dropout = dropout
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = 0.5
        self.info: dict = {}
        self.preprocess_cfg: dict = {}
        self.module: AbsaHead | None = None
        self.tokenizer = None

    def _encode(self, texts: list[str]) -> dict[str, Tensor]:
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

    def _outputs(self, texts: list[str]) -> dict[str, Tensor]:
        """Raw head outputs for all texts, on CPU."""
        self.module.eval()
        parts: list[dict[str, Tensor]] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.eval_batch_size):
                enc = self._encode(texts[i : i + self.eval_batch_size])
                out = self.module(
                    enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                )
                parts.append({k: v.float().cpu() for k, v in out.items()})
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def fit(
        self,
        texts: list[str],
        Y: np.ndarray,
        dev_texts: list[str] | None = None,
        Y_dev: np.ndarray | None = None,
    ) -> "TransformerAbsa":
        from transformers import AutoModel, AutoTokenizer

        from absa.tiny import build_tiny
        from absa.training import fit_loop, set_seed

        set_seed(self.seed)
        if self.model_name == "__tiny__":
            encoder, self.tokenizer = build_tiny(texts)
        else:
            encoder = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.module = AbsaHead(encoder, self.mode, self.dropout)
        weights = compute_class_weights(Y).to(self.device) if self.class_weight else None
        labels = torch.as_tensor(np.asarray(Y), dtype=torch.long)
        rng = np.random.default_rng(self.seed)

        def train_batches():
            order = rng.permutation(len(texts))
            for i in range(0, len(texts), self.batch_size):
                idx = order[i : i + self.batch_size]
                enc = self._encode([texts[j] for j in idx])
                yield {**enc, "labels": labels[idx]}

        def loss_fn(module: nn.Module, batch: dict) -> Tensor:
            out = module(
                batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
            )
            return absa_loss(out, batch["labels"].to(self.device), self.mode, weights)

        has_dev = dev_texts is not None and Y_dev is not None and len(dev_texts) > 0
        result = fit_loop(
            self.module,
            train_batches,
            (lambda: self.predict(dev_texts)) if has_dev else None,
            Y_dev if has_dev else None,
            loss_fn,
            epochs=self.epochs,
            lr=self.lr,
            patience=self.patience,
            fp16=self.fp16,
            device=self.device,
        )

        self.threshold = 0.5
        if self.mode == "split" and has_dev:
            out = self._outputs(dev_texts)
            self.threshold = best_threshold(
                torch.sigmoid(out["aspect_logits"]).numpy(),
                (out["polarity_logits"].argmax(-1) + 1).numpy(),
                np.asarray(Y_dev),
            )
        self.info = {**result, "threshold": self.threshold, "device": self.device}
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.module is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        if not texts:
            return np.zeros((0, N_ASPECTS), dtype=np.int8)
        return decode(self._outputs(texts), self.mode, self.threshold)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "model_type": "transformer",
            "model_name": self.model_name,
            "mode": self.mode,
            "max_length": self.max_length,
            "threshold": self.threshold,
            "dropout": self.dropout,
            "preprocess": self.preprocess_cfg,
        }
        (path / "absa_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        torch.save(self.module.state_dict(), path / "model.pt")
        # config.json lets Task 12 rebuild the encoder offline, without downloading weights
        self.module.encoder.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

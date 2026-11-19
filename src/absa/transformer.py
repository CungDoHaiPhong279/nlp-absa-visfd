"""Transformer ABSA core: output heads, losses, decoding, class weights, threshold search.

`TransformerAbsa` (training wrapper) is added in Task 9.
"""

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

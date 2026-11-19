import numpy as np
import pytest
import torch
import torch.nn.functional as F

from absa.data import ASPECTS
from absa.tiny import build_tiny
from absa.transformer import (
    AbsaHead,
    absa_loss,
    best_threshold,
    compute_class_weights,
    decode,
)

N = len(ASPECTS)
CAMERA = ASPECTS.index("CAMERA")
STORAGE = ASPECTS.index("STORAGE")


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    return build_tiny(["pin trâu", "máy đẹp quá", "giá cao"])


def _batch(tokenizer):
    return tokenizer(["pin trâu", "giá cao quá"], padding=True, return_tensors="pt")


def test_build_tiny_offline(tiny):
    encoder, tokenizer = tiny
    assert tokenizer(["pin trâu"])["input_ids"][0][0] == tokenizer.convert_tokens_to_ids("<s>")
    assert encoder.config.pad_token_id == tokenizer.pad_token_id


def test_joint_shapes(tiny):
    encoder, tokenizer = tiny
    batch = _batch(tokenizer)
    out = AbsaHead(encoder, "joint")(batch["input_ids"], batch["attention_mask"])
    assert out["logits"].shape == (2, N, 4)


def test_split_shapes(tiny):
    encoder, tokenizer = tiny
    batch = _batch(tokenizer)
    out = AbsaHead(encoder, "split")(batch["input_ids"], batch["attention_mask"])
    assert out["aspect_logits"].shape == (2, N)
    assert out["polarity_logits"].shape == (2, N, 3)


def _y() -> torch.Tensor:
    y = torch.zeros(2, N, dtype=torch.long)
    y[0, CAMERA] = 1
    y[1, 3] = 3
    return y


def test_joint_loss_with_weights() -> None:
    y = _y()
    out = {"logits": torch.randn(2, N, 4)}
    loss = absa_loss(out, y, "joint", class_weights=compute_class_weights(y.numpy()))
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_split_loss_ignores_absent_aspects() -> None:
    torch.manual_seed(1)
    y = _y()
    out = {"aspect_logits": torch.randn(2, N), "polarity_logits": torch.randn(2, N, 3)}
    base = absa_loss(out, y, "split")
    changed = out["polarity_logits"].clone()
    absent = y == 0
    changed[absent] = torch.randn(int(absent.sum()), 3) * 10
    other = absa_loss({**out, "polarity_logits": changed}, y, "split")
    assert torch.allclose(base, other)


def test_split_loss_no_present_aspect() -> None:
    y = torch.zeros(2, N, dtype=torch.long)
    out = {"aspect_logits": torch.randn(2, N), "polarity_logits": torch.randn(2, N, 3)}
    loss = absa_loss(out, y, "split")
    bce = F.binary_cross_entropy_with_logits(out["aspect_logits"], y.float())
    assert torch.isfinite(loss) and torch.allclose(loss, bce)


def test_split_class_weights_raises() -> None:
    y = _y()
    out = {"aspect_logits": torch.randn(2, N), "polarity_logits": torch.randn(2, N, 3)}
    with pytest.raises(ValueError):
        absa_loss(out, y, "split", class_weights=torch.ones(N, 4))


def test_decode_joint_argmax() -> None:
    logits = torch.zeros(1, N, 4)
    logits[0, CAMERA, 3] = 5.0
    pred = decode({"logits": logits}, "joint")
    assert pred.dtype == np.int8 and pred.shape == (1, N)
    assert pred[0, CAMERA] == 3 and pred.sum() == 3


def test_decode_split_threshold() -> None:
    aspect = torch.full((1, N), -5.0)
    aspect[0, CAMERA] = 5.0
    polarity = torch.zeros(1, N, 3)
    polarity[0, :, 2] = 9.0
    pred = decode({"aspect_logits": aspect, "polarity_logits": polarity}, "split")
    assert pred.dtype == np.int8
    assert pred[0, CAMERA] == 3
    assert (np.delete(pred[0], CAMERA) == 0).all()


def test_class_weights_zero_column() -> None:
    y = np.zeros((20, N), dtype=np.int8)
    y[:2, CAMERA] = 1  # CAMERA: 18 x None, 2 x Positive
    w = compute_class_weights(y)
    assert w.shape == (N, 4) and w.dtype == torch.float32
    assert torch.isfinite(w).all() and (w >= 0.1).all() and (w <= 10).all()
    assert y[:, STORAGE].max() == 0
    assert w[CAMERA, 0] < w[CAMERA, 1]


def test_best_threshold() -> None:
    y_true = np.zeros((2, N), dtype=np.int8)
    y_true[:, CAMERA] = 1
    probs = np.full((2, N), 0.1)
    probs[:, CAMERA] = 0.4
    codes = np.ones((2, N), dtype=np.int8)
    assert best_threshold(probs, codes, y_true) == 0.35

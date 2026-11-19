import json

import numpy as np
import pytest

from absa.data import labels_to_matrix
from absa.transformer import THRESHOLD_GRID, TransformerAbsa

TEXTS = [
    "pin trâu",
    "pin yếu",
    "máy đẹp",
    "giá cao",
    "camera nét",
    "giao hàng nhanh",
    "màn hình đẹp",
    "hihi",
]
LABELS = [
    {"BATTERY": "Positive"},
    {"BATTERY": "Negative"},
    {"DESIGN": "Positive"},
    {"PRICE": "Negative"},
    {"CAMERA": "Positive"},
    {"SER&ACC": "Positive"},
    {"SCREEN": "Positive"},
    {},
]
Y = labels_to_matrix(LABELS)


def _model(**kw) -> TransformerAbsa:
    return TransformerAbsa("__tiny__", epochs=1, batch_size=4, fp16=False, **kw)


def test_fit_predict_joint() -> None:
    model = _model().fit(TEXTS, Y)
    pred = model.predict(TEXTS)
    assert pred.shape == (8, 10) and pred.dtype == np.int8


def test_fit_predict_split() -> None:
    model = _model(mode="split").fit(TEXTS, Y, TEXTS[:4], Y[:4])
    pred = model.predict(TEXTS)
    assert pred.shape == (8, 10) and pred.dtype == np.int8
    assert model.info["threshold"] in THRESHOLD_GRID


def test_truncation_long_input() -> None:
    model = _model(max_length=256).fit(TEXTS, Y)
    assert model.predict([" ".join(["pin"] * 2000)]).shape == (1, 10)


def test_class_weight_split_raises() -> None:
    with pytest.raises(ValueError):
        TransformerAbsa("__tiny__", mode="split", class_weight=True)


def test_max_length_over_256_raises() -> None:
    with pytest.raises(ValueError):
        TransformerAbsa("__tiny__", max_length=257)


def test_save_writes_files(tmp_path) -> None:
    model = _model(mode="split").fit(TEXTS, Y, TEXTS[:4], Y[:4])
    model.preprocess_cfg = {"segment": None}
    model.save(tmp_path)
    for name in ("absa_config.json", "model.pt", "config.json", "tokenizer.json"):
        assert (tmp_path / name).is_file(), name
    cfg = json.loads((tmp_path / "absa_config.json").read_text(encoding="utf-8"))
    assert cfg["mode"] == "split" and cfg["threshold"] in THRESHOLD_GRID
    assert cfg["model_type"] == "transformer" and cfg["preprocess"] == {"segment": None}

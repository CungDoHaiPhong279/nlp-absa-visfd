"""Tests for the TF-IDF ABSA baseline."""

from pathlib import Path

import numpy as np
import pytest

from absa.baseline import TfidfAbsa
from absa.data import ASPECTS


def _labels(rows: list[dict[str, int]]) -> np.ndarray:
    y = np.zeros((len(rows), len(ASPECTS)), dtype=np.int8)
    for i, row in enumerate(rows):
        for aspect, code in row.items():
            y[i, ASPECTS.index(aspect)] = code
    return y


@pytest.mark.parametrize("mode", ["split", "joint"])
def test_predict_shape_dtype(mode: str) -> None:
    texts = ["pin tốt", "pin tệ", "camera đẹp", "giao nhanh"]
    y = _labels(
        [
            {"BATTERY": 1},
            {"BATTERY": 3},
            {"CAMERA": 1},
            {"SER&ACC": 1},
        ]
    )
    model = TfidfAbsa(mode=mode, min_df=1).fit(texts, y)
    pred = model.predict(["pin tốt", "giao nhanh"])
    assert pred.shape == (2, 10)
    assert pred.dtype == np.int8
    assert set(np.unique(pred)) <= {0, 1, 2, 3}


def test_vectorizer_fit_on_train_only() -> None:
    model = TfidfAbsa(min_df=1).fit(
        ["pin trâu", "pin yếu"], _labels([{"BATTERY": 1}, {"BATTERY": 3}])
    )
    vocabulary = model.vectorizer.vocabulary_.copy()
    model.predict(["camera đẹp"])
    assert model.vectorizer.vocabulary_ == vocabulary
    assert "camera" not in vocabulary


@pytest.mark.parametrize("mode", ["split", "joint"])
def test_single_class_aspect(mode: str) -> None:
    model = TfidfAbsa(mode=mode, min_df=1).fit(
        ["pin tốt", "pin yếu", "giá cao"],
        _labels([{"BATTERY": 1}, {"BATTERY": 3}, {"PRICE": 3}]),
    )
    assert np.all(model.predict(["pin tốt", "giá cao"])[:, ASPECTS.index("STORAGE")] == 0)


@pytest.mark.parametrize("mode", ["split", "joint"])
def test_single_polarity_aspect(mode: str) -> None:
    model = TfidfAbsa(mode=mode, min_df=1).fit(
        ["camera đẹp", "camera tốt", "pin yếu", "giá cao"],
        _labels([{"CAMERA": 1}, {"CAMERA": 1}, {"BATTERY": 3}, {"PRICE": 3}]),
    )
    predictions = model.predict(["camera đẹp", "camera tốt", "pin yếu"])
    assert set(predictions[:, ASPECTS.index("CAMERA")]) <= {0, 1}


@pytest.mark.parametrize("mode", ["split", "joint"])
@pytest.mark.parametrize("classifier", ["lr", "svm"])
def test_learns_trivial_signal(mode: str, classifier: str) -> None:
    texts = ["pin tốt"] * 10 + ["pin tệ"] * 10 + ["giao nhanh"] * 10
    y = _labels([{"BATTERY": 1}] * 10 + [{"BATTERY": 3}] * 10 + [{"SER&ACC": 1}] * 10)
    model = TfidfAbsa(mode=mode, classifier=classifier, min_df=1).fit(texts, y)
    assert model.predict(["pin tốt", "pin tệ"])[:, ASPECTS.index("BATTERY")].tolist() == [
        1,
        3,
    ]


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        TfidfAbsa(mode="x")


def test_run_experiment_smoke_real(tmp_path: Path) -> None:
    if not Path("data/raw/Train.csv").exists():
        pytest.skip("Dữ liệu UIT-ViSFD chưa được tải")
    experiment = pytest.importorskip("absa.experiment", reason="Task 5 chưa được tích hợp")
    config = pytest.importorskip("absa.config", reason="Task 5 chưa được tích hợp")
    result = experiment.run_experiment(
        config.load_config(Path("configs/smoke/tfidf_smoke.yaml")),
        limit=200,
        out_dir=tmp_path,
    )
    assert (tmp_path / "runs" / "tfidf_smoke__seed42.json").exists()
    assert result["test"] is None
    assert 0 <= result["dev"]["aspect_macro_f1"] <= 1

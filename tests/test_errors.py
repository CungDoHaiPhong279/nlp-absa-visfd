import numpy as np
import pandas as pd

from absa.data import format_label, labels_to_matrix
from absa.errors import (
    ERROR_TYPES,
    collect_errors,
    load_predictions,
    plot_confusions,
    sample_errors,
)


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"index": range(n), "comment": [f"c{i}" for i in range(n)], "n_star": 5})


def test_collect_errors_categories() -> None:
    gold = [{"CAMERA": "Positive", "PRICE": "Negative"}, {"BATTERY": "Positive"}]
    pred = [{"CAMERA": "Negative", "BATTERY": "Positive"}, {"BATTERY": "Positive"}]
    errors = collect_errors(_df(2), labels_to_matrix(gold), labels_to_matrix(pred))
    assert len(errors) == 1
    row = errors.iloc[0]
    assert row["missed"] == ["PRICE"]
    assert row["spurious"] == ["BATTERY"]
    assert row["wrong_polarity"] == ["CAMERA"]
    assert row["n_errors"] == 3
    assert row["gold"] == format_label(gold[0]) and row["pred"] == format_label(pred[0])
    for col in ("error_type", "cause", "fix"):
        assert row[col] == ""
    assert "khác" in ERROR_TYPES


def _fake_errors() -> pd.DataFrame:
    rows = []
    for i in range(30):
        group = i // 10
        rows.append(
            {
                "index": i,
                "missed": ["PRICE"] if group == 1 else [],
                "spurious": ["BATTERY"] if group == 2 else [],
                "wrong_polarity": ["CAMERA"] if group == 0 else [],
            }
        )
    return pd.DataFrame(rows)


def _group(row: pd.Series) -> str:
    if row["wrong_polarity"]:
        return "wrong_polarity"
    return "missed" if row["missed"] else "spurious"


def test_sample_errors_deterministic_and_stratified() -> None:
    errors = _fake_errors()
    a = sample_errors(errors, n=9, seed=42)
    b = sample_errors(errors, n=9, seed=42)
    assert len(a) == 9
    assert set(a.apply(_group, axis=1)) == {"wrong_polarity", "missed", "spurious"}
    assert a["index"].tolist() == b["index"].tolist()
    assert len(sample_errors(errors, n=100)) == 30


def test_load_predictions_roundtrip(tmp_path) -> None:
    gold = [{"CAMERA": "Positive"}, {}, {"PRICE": "Neutral", "SCREEN": "Negative"}]
    pred = [{"CAMERA": "Positive"}, {"BATTERY": "Negative"}, {}]
    path = tmp_path / "pred.csv"
    df = _df(3).assign(gold=[format_label(g) for g in gold], pred=[format_label(p) for p in pred])
    df.to_csv(path, index=False)
    out, Y_true, Y_pred = load_predictions(path)
    assert len(out) == 3
    assert np.array_equal(Y_true, labels_to_matrix(gold))
    assert np.array_equal(Y_pred, labels_to_matrix(pred))


def test_plot_confusions_writes_png(tmp_path) -> None:
    Y = labels_to_matrix([{"BATTERY": "Positive"}, {"STORAGE": "Negative"}, {}])
    path = tmp_path / "conf.png"
    plot_confusions(Y, Y, ["BATTERY", "STORAGE"], path)
    assert path.is_file() and path.stat().st_size > 0

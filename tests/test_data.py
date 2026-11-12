import re
from pathlib import Path

import numpy as np
import pytest

from absa.data import (
    ASPECTS,
    format_label,
    has_others,
    labels_to_matrix,
    load_split,
    matrix_to_labels,
    parse_label,
)


def test_aspect_order():
    assert ASPECTS == (
        "SCREEN",
        "CAMERA",
        "FEATURES",
        "BATTERY",
        "PERFORMANCE",
        "STORAGE",
        "DESIGN",
        "PRICE",
        "GENERAL",
        "SER&ACC",
    )


def test_parse_basic():
    assert parse_label("{CAMERA#Positive};{BATTERY#Negative};{OTHERS};") == {
        "CAMERA": "Positive",
        "BATTERY": "Negative",
    }


def test_parse_others_only():
    assert parse_label("{OTHERS};") == {}
    assert has_others("{OTHERS};") is True
    assert has_others("{CAMERA#Positive};") is False


def test_parse_whitespace_and_no_trailing_semicolon():
    assert parse_label(" {SER&ACC#Neutral} ; {PRICE#Positive}") == {
        "SER&ACC": "Neutral",
        "PRICE": "Positive",
    }


@pytest.mark.parametrize(
    "bad, token",
    [
        ("{CAMRA#Positive};", "CAMRA#Positive"),
        ("{CAMERA#Good};", "CAMERA#Good"),
        ("{CAMERA};", "CAMERA"),
        ("{CAMERA#Positive};{CAMERA#Negative};", "CAMERA"),
    ],
)
def test_parse_invalid_raises(bad, token):
    with pytest.raises(ValueError, match=re.escape(token)):
        parse_label(bad)


def test_format_label_order_and_empty():
    assert (
        format_label({"BATTERY": "Negative", "CAMERA": "Positive"})
        == "{CAMERA#Positive};{BATTERY#Negative};"
    )
    assert format_label({}) == ""


def test_matrix_roundtrip():
    labels = [{"CAMERA": "Positive", "STORAGE": "Negative"}, {}]
    Y = labels_to_matrix(labels)
    assert Y.shape == (2, 10) and Y.dtype == np.int8
    assert Y[0, ASPECTS.index("CAMERA")] == 1 and Y[0, ASPECTS.index("STORAGE")] == 3
    assert Y[1].sum() == 0
    assert matrix_to_labels(Y) == labels
    assert labels_to_matrix([]).shape == (0, 10)


def test_load_split_fake(fake_data_dir):
    df = load_split("train", fake_data_dir)
    assert len(df) == 12
    assert {"index", "comment", "n_star", "date_time", "label", "aspects", "has_others"} <= set(
        df.columns
    )
    assert df["n_star"].dtype.kind == "i"
    assert int(df["aspects"].map(len).eq(0).sum()) == 3


REAL = Path("data/raw/Train.csv")


@pytest.mark.skipif(not REAL.exists(), reason="chưa tải dữ liệu")
def test_load_split_real_sizes():
    sizes = {s: len(load_split(s)) for s in ("train", "dev", "test")}
    assert sizes == {"train": 7786, "dev": 1112, "test": 2224}
    only_others = {
        s: int(load_split(s)["aspects"].map(len).eq(0).sum()) for s in ("train", "dev", "test")
    }
    assert only_others == {"train": 115, "dev": 18, "test": 37}
    assert int(load_split("train")["aspects"].map(lambda d: "STORAGE" in d).sum()) == 91

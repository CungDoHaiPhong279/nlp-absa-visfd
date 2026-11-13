import pandas as pd
import pytest

from absa.data import ASPECTS, parse_label
from absa.eda import (
    aspect_polarity_table,
    aspects_per_comment,
    duplicate_report,
    emoji_stats,
    length_stats,
    star_vs_polarity,
    teencode_counts,
    write_json,
    write_table,
)


@pytest.fixture
def df():
    labels = ["{CAMERA#Positive};{PRICE#Negative};", "{OTHERS};", "{CAMERA#Neutral};"]
    return pd.DataFrame(
        {"label": labels, "n_star": [5, 1, 3], "aspects": [parse_label(x) for x in labels]}
    )


def df_of(texts):
    return pd.DataFrame({"comment": texts})


def test_aspect_polarity_table(df):
    t = aspect_polarity_table(df)
    assert list(t.index) == list(ASPECTS)
    assert list(t.columns) == ["Positive", "Neutral", "Negative", "Total"]
    assert t.loc["CAMERA", "Total"] == 2
    assert t.loc["PRICE", "Negative"] == 1
    assert t.loc["STORAGE", "Total"] == 0


def test_aspects_per_comment(df):
    assert aspects_per_comment(df).to_dict() == {0: 1, 1: 1, 2: 1}


def test_star_vs_polarity(df):
    t = star_vs_polarity(df)
    assert list(t.index) == [1, 2, 3, 4, 5]
    assert t.loc[5, "Positive"] == 1
    assert t.loc[5, "Negative"] == 1


def test_length_stats():
    s = length_stats(["a b c", "a"])
    assert s["syllables_mean"] == 2.0
    assert s["syllables_max"] == 3


def test_duplicate_report():
    splits = {"train": df_of(["a", "b", "A"]), "dev": df_of(["b"]), "test": df_of(["c"])}
    assert duplicate_report(splits) == {
        "train_dup": 1,
        "train_dev_overlap": 1,
        "train_test_overlap": 0,
        "dev_test_overlap": 0,
    }


def test_emoji_stats():
    s = emoji_stats(["pin 😍", "ok", "😍👍"])
    assert s["share_with_emoji"] == pytest.approx(2 / 3)
    assert s["top_emojis"][0] == ("😍", 2)


def test_teencode_counts():
    assert teencode_counts(["ko có gì", "Ko ok"])["ko"] == 2


def test_writers_use_lf_and_single_trailing_newline(tmp_path):
    write_table(pd.DataFrame({"a": [1, 2]}), tmp_path / "t.csv")
    write_json({"x": "é", "y": [1]}, tmp_path / "j.json")
    for name in ("t.csv", "j.json"):
        b = (tmp_path / name).read_bytes()
        assert b"\r" not in b
        assert b.endswith(b"\n") and not b.endswith(b"\n\n")

import json

from absa.data import ASPECTS
from absa.report import (
    ablation_table,
    collect_runs,
    comparison_table,
    per_aspect_table,
    sweep_table,
)


def _write_run(runs_dir, name, seed, score, groups=()):
    per_aspect = {
        aspect: {
            "precision": score,
            "recall": score,
            "f1": score,
            "support": 2,
        }
        for aspect in ASPECTS
    }
    metrics = {
        "aspect_macro_f1": score,
        "aspect_micro_f1": score,
        "sentiment_macro_f1": score,
        "sentiment_micro_f1": score,
        "polarity_macro_f1_detected": score,
        "exact_match": score,
        "per_aspect": per_aspect,
    }
    payload = {
        "name": name,
        "seed": seed,
        "groups": list(groups),
        "train_seconds": 1.0,
        "dev": metrics,
        "test": None,
    }
    path = runs_dir / f"{name}__seed{seed}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    _write_run(runs_dir, "phobert", 42, 0.80)
    _write_run(runs_dir, "phobert", 43, 0.82)
    _write_run(runs_dir, "tfidf", 42, 0.80, groups=["E1"])

    return collect_runs(runs_dir)


def test_comparison_table_averages_seeds_and_marks_published(tmp_path):
    runs = _make_runs(tmp_path)

    table = comparison_table(runs, "dev", published_metric=None)

    phobert = table.loc[table["name"] == "phobert"].iloc[0]
    tfidf = table.loc[table["name"] == "tfidf"].iloc[0]
    published = table.iloc[-1]

    assert phobert["aspect_macro_f1"] == "81.00 ± 1.41"
    assert phobert["n_seeds"] == 2
    assert tfidf["aspect_macro_f1"] == "80.00"
    assert tfidf["n_seeds"] == 1
    assert published["source"] == "published"
    assert "không cùng định nghĩa" in published["note"]


def test_comparison_table_test_has_only_published_when_no_test_runs(tmp_path):
    runs = _make_runs(tmp_path)

    table = comparison_table(runs, "test", published_metric=None)

    assert len(table) == 1
    assert table.iloc[0]["source"] == "published"


def test_ablation_table_filters_by_group(tmp_path):
    runs = _make_runs(tmp_path)

    table = ablation_table(runs, "E1")

    assert table["name"].tolist() == ["tfidf"]


def test_per_aspect_table_has_all_aspects(tmp_path):
    runs = _make_runs(tmp_path)

    table = per_aspect_table(runs, "dev", ["phobert", "tfidf"])

    assert len(table) == 10
    assert table.index.tolist() == list(ASPECTS)


def test_sweep_table_filters_by_prefix(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    _write_run(runs_dir, "tfidf_lr__C=0.25", 42, 0.30)
    _write_run(runs_dir, "tfidf_lr__C=0.5", 42, 0.35)
    _write_run(runs_dir, "tfidf_svm__C=0.25", 42, 0.45)

    runs = collect_runs(runs_dir)
    table = sweep_table(runs, prefix="tfidf_lr__C=")

    assert table["name"].tolist() == [
        "tfidf_lr__C=0.25",
        "tfidf_lr__C=0.5",
    ]

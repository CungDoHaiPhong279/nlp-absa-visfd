from pathlib import Path

import pytest

from absa.config import apply_overrides, load_config


def test_inherit_deep_merge(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "name: b\nseed: 42\ngroups: [E1]\n"
        "preprocess: {lowercase: true}\nmodel: {type: tfidf, C: 1}\n",
        encoding="utf-8",
    )
    child.write_text(
        "base: base.yaml\nname: c\nmodel: {C: 2}\n",
        encoding="utf-8",
    )

    cfg = load_config(child)

    assert cfg["model"] == {"type": "tfidf", "C": 2}
    assert cfg["preprocess"] == {"lowercase": True}
    assert cfg["groups"] == []
    assert "base" not in cfg


def test_missing_name_raises(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="name"):
        load_config(path)


def test_override_suffix_and_parse() -> None:
    cfg = {
        "name": "b",
        "groups": ["E1"],
        "model": {"C": 1},
        "preprocess": {"segment": "underthesea"},
    }

    changed = apply_overrides(cfg, ["model.C=0.5", "preprocess.segment=null"])

    assert changed["model"]["C"] == 0.5
    assert changed["preprocess"]["segment"] is None
    assert changed["name"] == "b__C=0.5__segment=null"
    assert changed["groups"] == []
    assert cfg["model"]["C"] == 1
    assert cfg["preprocess"]["segment"] == "underthesea"


def test_override_unknown_key_raises() -> None:
    cfg = {"name": "b", "groups": [], "model": {"C": 1}}

    with pytest.raises(ValueError, match="Cc"):
        apply_overrides(cfg, ["model.Cc=1"])


def test_override_scientific_notation_is_float() -> None:
    cfg = {"name": "b", "groups": [], "model": {"lr": 2.0e-5}}

    assert apply_overrides(cfg, ["model.lr=1e-5"])["model"]["lr"] == 1e-5

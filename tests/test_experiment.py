import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from absa import registry
from absa.config import assert_final_config
from absa.experiment import build_model, run_experiment


class DummyModel:
    instances = []

    def __init__(self, seed=42):
        self.seed = seed
        self.instances.append(self)

    def fit(self, texts, Y, dev_texts=None, Y_dev=None):
        self.fit_texts = list(texts)
        self.fit_dev_texts = list(dev_texts or [])
        return self

    def predict(self, texts):
        return np.zeros((len(texts), 10), dtype=np.int8)


def _config():
    return {
        "name": "d",
        "seed": 42,
        "preprocess": {},
        "model": {"type": "dummy"},
    }


def test_run_writes_json_and_predictions(fake_data_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, "dummy", DummyModel)

    result = run_experiment(
        _config(),
        data_dir=fake_data_dir,
        out_dir=tmp_path,
        cache_dir=None,
    )

    raw = (tmp_path / "runs" / "d__seed42.json").read_bytes()
    assert raw.endswith(b"}\n") and b"\r\n" not in raw
    saved = json.loads(raw.decode("utf-8"))
    predictions = pd.read_csv(tmp_path / "predictions" / "d__seed42__dev.csv")

    assert saved["test"] is None
    assert saved["dev"]["exact_match"] == 0.25
    assert result == saved
    assert list(predictions.columns) == ["index", "comment", "n_star", "gold", "pred"]
    assert len(predictions) == 4


def test_fit_never_sees_test(fake_data_dir: Path, tmp_path: Path, monkeypatch):
    DummyModel.instances.clear()
    monkeypatch.setitem(registry.REGISTRY, "dummy", DummyModel)

    result = run_experiment(
        _config(),
        eval_test=True,
        data_dir=fake_data_dir,
        out_dir=tmp_path,
        cache_dir=None,
    )

    model = DummyModel.instances[-1]
    assert all(not text.startswith("test ") for text in model.fit_texts + model.fit_dev_texts)
    assert result["test"] is not None


def test_limit_slices_train_dev(fake_data_dir: Path, tmp_path: Path, monkeypatch):
    DummyModel.instances.clear()
    monkeypatch.setitem(registry.REGISTRY, "dummy", DummyModel)

    run_experiment(
        _config(),
        limit=3,
        data_dir=fake_data_dir,
        out_dir=tmp_path,
        cache_dir=None,
    )

    model = DummyModel.instances[-1]
    assert len(model.fit_texts) == 3
    assert len(model.fit_dev_texts) == 3


def test_limit_run_never_saves_model(fake_data_dir: Path, tmp_path: Path, monkeypatch):
    class SaveableDummyModel(DummyModel):
        def save(self, path: Path) -> None:
            raise AssertionError("a --limit run must not overwrite the checkpoint")

    monkeypatch.setitem(registry.REGISTRY, "dummy", SaveableDummyModel)
    monkeypatch.chdir(tmp_path)
    cfg = _config()
    cfg["save"] = True
    run_experiment(
        cfg, limit=4, data_dir=fake_data_dir, out_dir=tmp_path / "results", cache_dir=None
    )
    assert not Path("models").exists()


def test_saves_model_when_config_requests_it(fake_data_dir: Path, tmp_path: Path, monkeypatch):
    class SaveableDummyModel(DummyModel):
        def save(self, path: Path) -> None:
            self.saved_path = path
            path.mkdir(parents=True, exist_ok=True)
            (path / "saved.txt").write_text("ok", encoding="utf-8")

    SaveableDummyModel.instances.clear()
    monkeypatch.setitem(registry.REGISTRY, "dummy", SaveableDummyModel)
    monkeypatch.chdir(tmp_path)
    cfg = _config()
    cfg["save"] = True
    cfg["preprocess"] = {"lowercase": True}

    run_experiment(
        cfg,
        data_dir=fake_data_dir,
        out_dir=tmp_path / "results",
        cache_dir=None,
    )

    model = SaveableDummyModel.instances[-1]
    expected_path = Path("models/d__seed42")
    assert model.saved_path == expected_path
    assert model.preprocess_cfg == {"lowercase": True}
    assert (expected_path / "saved.txt").read_text(encoding="utf-8") == "ok"


def test_eval_test_with_limit_raises(fake_data_dir: Path):
    with pytest.raises(ValueError, match="Không đánh giá test khi dùng --limit"):
        run_experiment(_config(), eval_test=True, limit=3, data_dir=fake_data_dir)


def test_eval_test_requires_final_config():
    with pytest.raises(ValueError):
        assert_final_config(Path("configs/tfidf_lr_base.yaml"))

    assert_final_config(Path("configs/final/x.yaml"))


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="tfidf"):
        build_model({"type": "abc"}, 42)


def test_cli_rejects_eval_test_on_non_final_config(tmp_path: Path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run.py",
            "--config",
            "configs/tfidf_lr_base.yaml",
            "--eval-test",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
    )
    assert proc.returncode != 0
    assert "configs/final" in proc.stderr
    assert not (tmp_path / "runs").exists()

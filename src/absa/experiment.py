import json
import platform
import random
import subprocess
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from absa.data import format_label, labels_to_matrix, load_split, matrix_to_labels
from absa.evaluate import evaluate
from absa.preprocess import preprocess_df
from absa.registry import Model, build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        from absa.training import set_seed
    except ImportError:
        return
    set_seed(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _evaluate_split(model: Model, frame: pd.DataFrame, texts: list[str]):
    y_true = labels_to_matrix(frame["aspects"].tolist())
    y_pred = model.predict(texts)

    metrics = evaluate(y_true, y_pred)
    gold = matrix_to_labels(y_true)
    pred = matrix_to_labels(y_pred)
    predictions = pd.DataFrame(
        {
            "index": frame["index"].tolist(),
            "comment": frame["comment"].tolist(),
            "n_star": frame["n_star"].tolist(),
            "gold": [format_label(row) for row in gold],
            "pred": [format_label(row) for row in pred],
        }
    )
    return metrics, predictions


def run_experiment(
    cfg: dict,
    *,
    eval_test: bool = False,
    limit: int | None = None,
    data_dir: Path = Path("data/raw"),
    out_dir: Path = Path("results"),
    cache_dir: Path | None = Path("data/processed"),
) -> dict:
    if eval_test and limit is not None:
        raise ValueError("Không đánh giá test khi dùng --limit")

    seed = int(cfg.get("seed", 42))
    splits = ("train", "dev", "test") if eval_test else ("train", "dev")
    frames = {split: load_split(split, data_dir=data_dir) for split in splits}
    texts = {
        split: preprocess_df(
            frame,
            cfg.get("preprocess", {}),
            cache_dir=cache_dir,
            cache_tag=split,
        )
        for split, frame in frames.items()
    }

    if limit is not None:
        for split in ("train", "dev"):
            frames[split] = frames[split].iloc[:limit]
            texts[split] = texts[split][:limit]

    _set_seed(seed)
    model = build_model(cfg["model"], seed)
    y_train = labels_to_matrix(frames["train"]["aspects"].tolist())
    y_dev = labels_to_matrix(frames["dev"]["aspects"].tolist())

    start = time.perf_counter()
    model.fit(
        texts["train"],
        y_train,
        dev_texts=texts["dev"],
        Y_dev=y_dev,
    )
    train_seconds = time.perf_counter() - start

    dev_metrics, dev_predictions = _evaluate_split(model, frames["dev"], texts["dev"])
    test_metrics = None
    test_predictions = None
    if eval_test:
        test_metrics, test_predictions = _evaluate_split(model, frames["test"], texts["test"])

    name = cfg["name"]
    # never save a --limit run: it would overwrite the full checkpoint under the same name
    if cfg.get("save") and limit is None and hasattr(model, "save"):
        model_dir = Path("models") / f"{name}__seed{seed}"
        model.preprocess_cfg = cfg.get("preprocess", {})
        model.save(model_dir)

    result = {
        "name": name,
        "seed": seed,
        "groups": cfg.get("groups", []),
        "config": cfg,
        "dev": dev_metrics,
        "test": test_metrics,
        "train_seconds": train_seconds,
        "limit": limit,
        "model_info": getattr(model, "info", {}),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().astimezone().isoformat(),
        "versions": _versions(),
    }

    runs_dir = Path(out_dir) / "runs"
    predictions_dir = Path(out_dir) / "predictions"
    runs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{name}__seed{seed}"
    (runs_dir / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    dev_predictions.to_csv(
        predictions_dir / f"{stem}__dev.csv",
        index=False,
        encoding="utf-8",
    )
    if test_predictions is not None:
        test_predictions.to_csv(
            predictions_dir / f"{stem}__test.csv",
            index=False,
            encoding="utf-8",
        )

    return result


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "torch", "transformers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result

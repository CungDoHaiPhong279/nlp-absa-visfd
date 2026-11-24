"""Collect and format ABSA experiment results."""

import json
from pathlib import Path
from typing import Literal

import pandas as pd

from absa.data import ASPECTS

MAIN_METRICS = (
    "aspect_macro_f1",
    "sentiment_macro_f1",
    "polarity_macro_f1_detected",
    "exact_match",
)

PUBLISHED = {
    "BiLSTM+fastText (Phan et al., 2021)": {
        "aspect": 84.48,
        "sentiment": 63.06,
    }
}


def collect_runs(runs_dir: Path = Path("results/runs")) -> pd.DataFrame:
    columns = [
        "name",
        "seed",
        "split",
        "groups",
        "train_seconds",
        *MAIN_METRICS,
        "per_aspect",
    ]
    rows = []

    # case-insensitive so Windows and Linux produce the same row order
    for path in sorted(Path(runs_dir).glob("*.json"), key=lambda p: p.name.lower()):
        run = json.loads(path.read_text(encoding="utf-8"))

        for split in ("dev", "test"):
            result = run.get(split)
            if result is None:
                continue

            row = {
                "name": run["name"],
                "seed": run["seed"],
                "split": split,
                "groups": run.get("groups", []),
                "train_seconds": run.get("train_seconds"),
                "per_aspect": result.get("per_aspect", {}),
            }
            row.update({metric: result.get(metric) for metric in MAIN_METRICS})
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def comparison_table(
    runs: pd.DataFrame,
    split: Literal["dev", "test"],
    names: list[str] | None = None,
    published_metric: dict[str, str] | None = None,
) -> pd.DataFrame:
    selected = runs.loc[runs["split"] == split].copy()

    if names is not None:
        selected = selected.loc[selected["name"].isin(names)]

    rows = []

    for name, group in selected.groupby("name", sort=False):
        row = {"name": name}

        for metric in MAIN_METRICS:
            values = group[metric].dropna().astype(float)

            if values.empty:
                row[metric] = "—"
            elif len(values) == 1:
                row[metric] = f"{values.mean() * 100:.2f}"
            else:
                mean = values.mean() * 100
                std = values.std(ddof=1) * 100
                row[metric] = f"{mean:.2f} ± {std:.2f}"

        row["n_seeds"] = group["seed"].nunique()
        row["source"] = "ours"
        row["note"] = ""
        rows.append(row)

    if names is not None:  # rows follow the caller's order (e.g. C ascending in sweeps)
        rows.sort(key=lambda r: names.index(r["name"]))

    published_name, published_values = next(iter(PUBLISHED.items()))
    published_row = {
        "name": published_name,
        **{metric: "—" for metric in MAIN_METRICS},
        "n_seeds": "—",
        "source": "published",
        "note": "",
    }

    if published_metric is None:
        published_row["note"] = "không cùng định nghĩa: 84.48 / 63.06"
    else:
        for label, metric in published_metric.items():
            published_row[metric] = f"{published_values[label]:.2f}"

    rows.append(published_row)

    columns = ["name", *MAIN_METRICS, "n_seeds", "source", "note"]
    return pd.DataFrame(rows, columns=columns)


def ablation_table(
    runs: pd.DataFrame,
    group: str,
    split: str = "dev",
) -> pd.DataFrame:
    selected = runs.loc[runs["groups"].map(lambda groups: group in groups)].copy()

    table = comparison_table(selected, split)
    return table.loc[table["source"] == "ours"].reset_index(drop=True)


def sweep_table(
    runs: pd.DataFrame,
    prefix: str,
    split: str = "dev",
) -> pd.DataFrame:
    selected = runs.loc[
        (runs["split"] == split) & runs["name"].astype(str).str.startswith(prefix)
    ].copy()

    names = sorted(
        selected["name"].drop_duplicates(),
        key=lambda name: float(name[len(prefix) :].split("__")[0]),
    )
    table = comparison_table(selected, split, names=names)

    return table.loc[table["source"] == "ours"].reset_index(drop=True)


def per_aspect_table(
    runs: pd.DataFrame,
    split: str,
    names: list[str],
) -> pd.DataFrame:
    selected = runs.loc[runs["split"] == split]
    rows = []

    for aspect in ASPECTS:
        row = {}
        supports = []

        for name in names:
            per_seed = selected.loc[selected["name"] == name, "per_aspect"]

            f1_values = []
            for scores in per_seed:
                result = scores.get(aspect)
                if result is None:
                    continue

                f1_values.append(float(result["f1"]))
                if "support" in result:
                    supports.append(int(result["support"]))

            if f1_values:
                mean_f1 = sum(f1_values) / len(f1_values) * 100
                row[name] = f"{mean_f1:.2f}"
            else:
                row[name] = "—"

        # Support là số mẫu thật của khía cạnh trên split, không cộng
        # lặp lại qua các seed.
        row["support"] = supports[0] if supports else 0
        rows.append(row)

    table = pd.DataFrame(rows, index=list(ASPECTS), columns=[*names, "support"])
    table.index.name = "aspect"
    return table

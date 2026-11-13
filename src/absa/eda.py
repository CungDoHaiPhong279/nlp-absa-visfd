"""Exploratory data analysis helpers for UIT-ViSFD (tables, stats and figures)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from absa.data import ASPECTS, POLARITIES  # noqa: E402
from absa.preprocess import _EMOJI, normalize  # noqa: E402

TEENCODE: tuple[str, ...] = (
    "k", "ko", "kh", "hk", "khg", "dc", "đc", "ok", "oke", "sp", "mn", "bt",
    "vs", "j", "r", "mik", "mk", "đt", "dt", "cam", "fb", "nv", "trc", "ms",
)  # fmt: skip
COLORS = {"Positive": "#1a7f37", "Neutral": "#8c959f", "Negative": "#cf222e"}
_EMOJI_MODIFIERS = set("️‍⃣")  # variation selector, ZWJ, keycap: not emoji themselves


def aspect_polarity_table(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(0, index=list(ASPECTS), columns=list(POLARITIES), dtype=int)
    for d in df["aspects"]:
        for a, p in d.items():
            t.loc[a, p] += 1
    t["Total"] = t.sum(axis=1)
    return t


def aspects_per_comment(df: pd.DataFrame) -> pd.Series:
    return df["aspects"].map(len).value_counts().sort_index()


def star_vs_polarity(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(0, index=range(1, 6), columns=list(POLARITIES), dtype=int)
    t.index.name = "n_star"
    for star, d in zip(df["n_star"], df["aspects"], strict=True):
        for p in d.values():
            t.loc[star, p] += 1
    return t


def length_stats(texts: list[str]) -> dict:
    syl = np.array([len(t.split()) for t in texts])
    return {
        "chars_mean": float(np.mean([len(t) for t in texts])),
        "syllables_mean": float(syl.mean()),
        "syllables_p95": float(np.percentile(syl, 95)),
        "syllables_p99": float(np.percentile(syl, 99)),
        "syllables_max": int(syl.max()),
    }


def duplicate_report(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    norm = {k: [normalize(str(c), lowercase=True) for c in v["comment"]] for k, v in splits.items()}
    s = {k: set(v) for k, v in norm.items()}
    return {
        "train_dup": len(norm["train"]) - len(s["train"]),
        "train_dev_overlap": len(s["train"] & s["dev"]),
        "train_test_overlap": len(s["train"] & s["test"]),
        "dev_test_overlap": len(s["dev"] & s["test"]),
    }


def emoji_stats(texts: list[str], top: int = 10) -> dict:
    found = [[c for c in _EMOJI.findall(t) if c not in _EMOJI_MODIFIERS] for t in texts]
    counts = Counter(c for f in found for c in f)
    return {
        "share_with_emoji": sum(bool(f) for f in found) / len(texts),
        "top_emojis": counts.most_common(top),
    }


def teencode_counts(texts: list[str], terms: tuple[str, ...] = TEENCODE) -> pd.Series:
    bags = [set(t.lower().split()) for t in texts]
    s = pd.Series({w: sum(w in b for b in bags) for w in terms}, dtype=int)
    return s.sort_values(ascending=False, kind="stable")


def subword_lengths(texts: list[str], tokenizer) -> np.ndarray:
    """Subword counts including special tokens, no truncation."""
    enc = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    return np.array([len(x) for x in enc])


def subword_row(lengths: np.ndarray) -> dict:
    return {
        "mean": float(lengths.mean()),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }


def write_table(df: pd.DataFrame | pd.Series, path: Path, **kw) -> None:
    """CSV with LF endings on every OS, so pre-commit never rewrites it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, lineterminator="\n", **kw)


def write_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _save(fig, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aspect_freq(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(tables)
    x = np.arange(len(ASPECTS))
    for i, (name, t) in enumerate(tables.items()):
        ax.bar(x + i * w, t["Total"].values, w, label=name)
    ax.set_xticks(x + w * (len(tables) - 1) / 2, ASPECTS, rotation=45, ha="right")
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Số lượt gán nhãn")
    ax.legend(title="Tập")
    _save(fig, path)


def plot_aspect_polarity_stacked(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(table))
    for p in POLARITIES:
        ax.bar(table.index, table[p].values, bottom=bottom, color=COLORS[p], label=p)
        bottom += table[p].values
    ax.set_xticks(range(len(table)), table.index, rotation=45, ha="right")
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Số lượt gán nhãn (Train)")
    ax.legend(title="Cảm xúc")
    _save(fig, path)


def plot_star_heatmap(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(table.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, int(table.values[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Cảm xúc")
    ax.set_ylabel("Số sao")
    fig.colorbar(im, ax=ax, label="Số cặp (khía cạnh, cảm xúc)")
    _save(fig, path)


def plot_length_hist(texts: list[str], path: Path) -> None:
    syl = [len(t.split()) for t in texts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(syl, bins=range(0, 201, 5), color="#0969da")
    ax.axvline(np.percentile(syl, 95), color="#cf222e", ls="--", label="p95")
    ax.set_xlabel("Số âm tiết mỗi bình luận (Train)")
    ax.set_ylabel("Số bình luận")
    ax.legend()
    _save(fig, path)

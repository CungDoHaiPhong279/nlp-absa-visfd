"""Hằng số, tách nhãn, ma trận nhãn và đọc split của UIT-ViSFD."""
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ASPECTS = ("SCREEN", "CAMERA", "FEATURES", "BATTERY", "PERFORMANCE",
           "STORAGE", "DESIGN", "PRICE", "GENERAL", "SER&ACC")
POLARITIES = ("Positive", "Neutral", "Negative")
POLARITY_CODE = {"Positive": 1, "Neutral": 2, "Negative": 3}
_CODE_POLARITY = {v: k for k, v in POLARITY_CODE.items()}
_TOKEN = re.compile(r"\{([^}]*)\}")
_COLUMNS = ("index", "comment", "n_star", "date_time", "label")


def _tokens(label: str) -> list[str]:
    return [t.strip() for t in _TOKEN.findall(label)]


def parse_label(label: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in _tokens(label):
        if tok == "OTHERS":
            continue
        aspect, sep, polarity = tok.partition("#")
        aspect, polarity = aspect.strip(), polarity.strip()
        if not sep or aspect not in ASPECTS or polarity not in POLARITIES:
            raise ValueError(f"Nhãn không hợp lệ: {tok}")
        if aspect in out:
            raise ValueError(f"Khía cạnh lặp: {aspect}")
        out[aspect] = polarity
    return out


def has_others(label: str) -> bool:
    return "OTHERS" in _tokens(label)


def format_label(aspects: dict[str, str]) -> str:
    return "".join(f"{{{a}#{aspects[a]}}};" for a in ASPECTS if a in aspects)


def labels_to_matrix(labels: list[dict[str, str]]) -> np.ndarray:
    Y = np.zeros((len(labels), len(ASPECTS)), dtype=np.int8)
    for i, d in enumerate(labels):
        for a, p in d.items():
            Y[i, ASPECTS.index(a)] = POLARITY_CODE[p]
    return Y


def matrix_to_labels(Y: np.ndarray) -> list[dict[str, str]]:
    return [{a: _CODE_POLARITY[int(c)] for a, c in zip(ASPECTS, row) if c} for row in Y]


def load_split(name: Literal["train", "dev", "test"], data_dir: Path = Path("data/raw")) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / f"{name.capitalize()}.csv", encoding="utf-8")
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    df["n_star"] = df["n_star"].astype(int)
    df["label"] = df["label"].astype(str)
    df["aspects"] = df["label"].map(parse_label)
    df["has_others"] = df["label"].map(has_others)
    return df

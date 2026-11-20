"""Error analysis helpers: collect, sample and plot model errors (Task 11)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from absa.data import ASPECTS, format_label, labels_to_matrix, matrix_to_labels, parse_label  # noqa: E402
from absa.evaluate import confusion_by_aspect  # noqa: E402

ERROR_TYPES = (
    "khía cạnh ngầm",
    "khen chê lẫn lộn",
    "mỉa mai",
    "phủ định",
    "teencode/viết tắt",
    "số sao mâu thuẫn",
    "nghi nhãn sai",
    "khác",
)
GROUPS = ("wrong_polarity", "missed", "spurious")
CLASS_NAMES = ("None", "Pos", "Neu", "Neg")


def load_predictions(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # keep_default_na=False: an empty label ({OTHERS}-only comment) must stay "", not NaN
    df = pd.read_csv(path, keep_default_na=False)
    Y_true = labels_to_matrix([parse_label(s) for s in df["gold"]])
    Y_pred = labels_to_matrix([parse_label(s) for s in df["pred"]])
    return df, Y_true, Y_pred


def collect_errors(df: pd.DataFrame, Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    gold, pred = matrix_to_labels(Y_true), matrix_to_labels(Y_pred)
    rows = []
    for i in range(len(Y_true)):
        t, p = Y_true[i], Y_pred[i]
        missed = [a for j, a in enumerate(ASPECTS) if t[j] and not p[j]]
        spurious = [a for j, a in enumerate(ASPECTS) if p[j] and not t[j]]
        wrong = [a for j, a in enumerate(ASPECTS) if t[j] and p[j] and t[j] != p[j]]
        n_errors = len(missed) + len(spurious) + len(wrong)
        if not n_errors:
            continue
        rows.append(
            {
                "index": df["index"].iloc[i],
                "comment": df["comment"].iloc[i],
                "n_star": df["n_star"].iloc[i],
                "gold": format_label(gold[i]),
                "pred": format_label(pred[i]),
                "missed": missed,
                "spurious": spurious,
                "wrong_polarity": wrong,
                "n_errors": n_errors,
                "error_type": "",
                "cause": "",
                "fix": "",
            }
        )
    columns = ["index", "comment", "n_star", "gold", "pred", "missed", "spurious"]
    columns += ["wrong_polarity", "n_errors", "error_type", "cause", "fix"]
    return pd.DataFrame(rows, columns=columns)


def _main_group(row: pd.Series) -> str:
    return next((g for g in GROUPS if len(row[g])), GROUPS[-1])


def sample_errors(errors: pd.DataFrame, n: int = 40, seed: int = 42) -> pd.DataFrame:
    if len(errors) <= n:
        return errors.copy()
    groups = errors.apply(_main_group, axis=1)
    sizes = {g: int((groups == g).sum()) for g in GROUPS if (groups == g).any()}
    # proportional quotas, at least 1 per non-empty group, largest remainder fills the rest
    exact = {g: n * s / len(errors) for g, s in sizes.items()}
    quota = {g: max(1, int(exact[g])) for g in sizes}
    for g in sorted(sizes, key=lambda g: exact[g] - int(exact[g]), reverse=True):
        if sum(quota.values()) >= n:
            break
        if quota[g] < sizes[g]:
            quota[g] += 1
    while sum(quota.values()) > n:
        quota[max(quota, key=quota.get)] -= 1
    parts = [
        errors[groups == g].sample(n=quota[g], random_state=seed) for g in GROUPS if g in quota
    ]
    return pd.concat(parts).sort_index()


def plot_confusions(Y_true: np.ndarray, Y_pred: np.ndarray, aspects: list[str], path: Path) -> None:
    cols = min(len(aspects), 3)
    rows = (len(aspects) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, aspect in zip(axes.flat, aspects, strict=False):
        cm = confusion_by_aspect(Y_true, Y_pred, aspect)
        ax.imshow(cm, cmap="Blues")
        for r in range(4):
            for c in range(4):
                color = "white" if cm[r, c] > cm.max() / 2 else "black"
                ax.text(c, r, int(cm[r, c]), ha="center", va="center", color=color)
        ax.set_xticks(range(4), CLASS_NAMES)
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Đúng")
        support = int((np.asarray(Y_true)[:, ASPECTS.index(aspect)] > 0).sum())
        ax.set_title(f"{aspect} (support={support})")
    for ax in list(axes.flat)[len(aspects) :]:
        ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)

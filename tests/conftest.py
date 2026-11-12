import pandas as pd
import pytest

_PATTERNS = [
    ("pin trâu máy đẹp", "{BATTERY#Positive};{DESIGN#Positive};"),
    ("pin yếu giá cao", "{BATTERY#Negative};{PRICE#Negative};"),
    ("giao hàng nhanh", "{SER&ACC#Positive};"),
    ("hihi", "{OTHERS};"),
]
_STARS = [5, 1, 4, 3]


@pytest.fixture
def fake_data_dir(tmp_path):
    for name, n in (("Train", 12), ("Dev", 4), ("Test", 4)):
        rows = []
        for i in range(n):
            text, label = _PATTERNS[i % 4]
            rows.append(
                {
                    "index": i,
                    "comment": f"{name.lower()} {text} {i}",
                    "n_star": _STARS[i % 4],
                    "date_time": "01/01/2021 10:00",
                    "label": label,
                }
            )
        pd.DataFrame(rows).to_csv(tmp_path / f"{name}.csv", index=False, encoding="utf-8")
    return tmp_path

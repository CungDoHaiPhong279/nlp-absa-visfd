import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data/raw/Train.csv").exists(), reason="chưa tải dữ liệu")
def test_smoke_pipeline(tmp_path) -> None:
    subprocess.run(
        [sys.executable, "scripts/run_all.py", "--smoke", "--workdir", str(tmp_path)],
        check=True,
        timeout=900,
        cwd=ROOT,
    )
    table = (tmp_path / "results/tables/comparison_dev.md").read_text(encoding="utf-8")
    assert "tfidf_smoke" in table and "tiny_transformer" in table

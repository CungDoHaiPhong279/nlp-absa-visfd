"""Chạy lại toàn bộ phần CPU của dự án (hoặc bản smoke) bằng một lệnh."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TFIDF_CONFIGS = [
    "configs/tfidf_lr_base.yaml",
    "configs/tfidf_svm.yaml",
    "configs/ablation/E1_char_25.yaml",
    "configs/ablation/E1_ngram_1.yaml",
    "configs/ablation/E1_ngram_13.yaml",
    "configs/ablation/E2_word.yaml",
    "configs/ablation/E3_tfidf_strip_emoji.yaml",
    "configs/ablation/E6_tfidf_joint.yaml",
]
SMOKE_CONFIGS = ["configs/smoke/tfidf_smoke.yaml", "configs/smoke/tiny_transformer.yaml"]
GPU_NOTE = """
Phần GPU (PhoBERT, ViSoBERT, Bi-LSTM, các cấu hình final) chạy trên Colab:
  mở notebooks/02_colab_train.ipynb (GPU T4) cho PhoBERT/ViSoBERT; Bi-LSTM và
  cấu hình final dùng các lệnh ở README mục 5, trong cùng phiên Colab.
  Chép results/runs/*.json về repo rồi chạy: python scripts/make_tables.py
"""


def step(name: str, args: list[str]) -> None:
    print(f"==> {name}", flush=True)
    start = time.perf_counter()
    try:
        subprocess.run([sys.executable, *args], check=True, cwd=ROOT)
    except subprocess.CalledProcessError:
        print(f"LỖI ở bước: {name}", file=sys.stderr)
        sys.exit(1)
    print(f"    xong {name} sau {time.perf_counter() - start:.0f} s", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1258 không in được tiếng Việt
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="chạy nhanh trên 200 dòng, CPU")
    parser.add_argument(
        "--workdir",
        type=Path,
        help="thư mục ghi results/ và data/processed/ (notebook EDA và data/raw vẫn ở repo)",
    )
    args = parser.parse_args()

    work = args.workdir.resolve() if args.workdir else ROOT
    results, cache = work / "results", work / "data/processed"
    out = ["--out-dir", str(results), "--cache-dir", str(cache)]

    step("Tải dữ liệu", ["scripts/download_data.py"])
    if args.smoke:
        for cfg in SMOKE_CONFIGS:
            step(f"Chạy {cfg}", ["scripts/run.py", "--config", cfg, "--limit", "200", *out])
    else:
        # notebook EDA luôn ghi bảng/hình vào results/ của repo
        step(
            "Notebook EDA",
            ["-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/01_eda.ipynb",
             "--output-dir", str(work / "notebooks_out")],
        )  # fmt: skip
        for cfg in TFIDF_CONFIGS:
            step(f"Chạy {cfg}", ["scripts/run.py", "--config", cfg, *out])
    step(
        "Tạo bảng",
        [
            "scripts/make_tables.py",
            "--runs",
            str(results / "runs"),
            "--out",
            str(results / "tables"),
        ],
    )
    if not args.smoke:
        print(GPU_NOTE)


if __name__ == "__main__":
    main()

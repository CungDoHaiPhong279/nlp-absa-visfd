import argparse
import re
from pathlib import Path

import yaml

from absa.report import (
    ablation_table,
    collect_runs,
    comparison_table,
    per_aspect_table,
    sweep_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tạo bảng từ kết quả các lần chạy.")
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    return parser


def read_published_metric(path: Path) -> dict[str, str] | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```yaml\s*(.*?)```", text, flags=re.DOTALL)

    if match is None:
        return None

    metadata = yaml.safe_load(match.group(1)) or {}
    return metadata.get("published_metric")


def write_table(table, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_aspect_table lưu khía cạnh ở index; đưa index thành cột
    # để khi xuất bảng không bị mất tên khía cạnh.
    output = table.reset_index() if table.index.name else table

    output.to_csv(
        out_dir / f"{name}.csv",
        index=False,
        lineterminator="\n",
    )
    markdown = output.to_markdown(index=False)
    (out_dir / f"{name}.md").write_text(
        markdown + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = build_parser().parse_args()

    runs = collect_runs(args.runs)
    published_metric = read_published_metric(Path("docs/metric_notes.md"))

    write_table(
        comparison_table(runs, "dev", published_metric=published_metric),
        args.out,
        "comparison_dev",
    )

    test_runs = runs.loc[runs["split"] == "test"]
    if not test_runs.empty:
        write_table(
            comparison_table(runs, "test", published_metric=published_metric),
            args.out,
            "comparison_test",
        )

        test_names = test_runs["name"].drop_duplicates().tolist()
        write_table(
            per_aspect_table(runs, "test", test_names),
            args.out,
            "per_aspect_test",
        )

    dev_runs = runs.loc[runs["split"] == "dev"]
    groups = sorted({group for run_groups in dev_runs["groups"] for group in run_groups})

    for group in groups:
        write_table(
            ablation_table(runs, group, split="dev"),
            args.out,
            f"ablation_{group}_dev",
        )

    sweeps = (
        ("tfidf_lr", "tfidf_lr__C="),
        ("tfidf_svm", "tfidf_svm__C="),
    )
    for model_name, prefix in sweeps:
        has_runs = dev_runs["name"].astype(str).str.startswith(prefix).any()
        if has_runs:
            write_table(
                sweep_table(runs, prefix, split="dev"),
                args.out,
                f"sweep_E0_{model_name}_dev",
            )


if __name__ == "__main__":
    main()

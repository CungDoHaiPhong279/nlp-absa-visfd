import argparse
from pathlib import Path

from absa.config import apply_overrides, assert_final_config, load_config
from absa.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    configs = []
    for filename in args.config:
        config = load_config(Path(filename))
        if args.overrides:
            config = apply_overrides(config, args.overrides)
        configs.append((Path(filename), config))

    # Chặn trước khi chạy bất kỳ config nào nếu yêu cầu đánh giá Test.
    if args.eval_test:
        for filename, _ in configs:
            assert_final_config(filename)

    for _, config in configs:
        seeds = args.seeds or [int(config.get("seed", 42))]
        for seed in seeds:
            run_config = dict(config)
            run_config["seed"] = seed
            result = run_experiment(
                run_config,
                eval_test=args.eval_test,
                limit=args.limit,
            )
            dev = result["dev"]
            print(
                f"{run_config['name']} seed={seed} "
                f"dev aspect={dev['aspect_macro_f1']:.2%} "
                f"sentiment={dev['sentiment_macro_f1']:.2%} "
                f"({result['train_seconds']:.0f}s)"
            )


if __name__ == "__main__":
    main()

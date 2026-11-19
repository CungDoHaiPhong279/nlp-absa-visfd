"""Filter fastText cc.vi.300 vectors down to the Train vocabulary of a Bi-LSTM config.

Run on Colab (the .vec.gz file is about 1.2 GB):
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz
    python scripts/prepare_fasttext.py --vec cc.vi.300.vec.gz --config configs/bilstm_fasttext.yaml
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from absa.bilstm import build_vocab, load_fasttext
from absa.config import load_config
from absa.data import load_split
from absa.preprocess import preprocess_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    texts = preprocess_df(
        load_split("train", data_dir=args.data_dir),
        cfg.get("preprocess", {}),
        cache_dir=args.cache_dir,
        cache_tag="train",
    )
    vocab = build_vocab(texts)
    matrix, found = load_fasttext(args.vec, vocab)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "fasttext_vi_train_vocab"
    np.save(stem.with_suffix(".npy"), matrix)
    stem.with_suffix(".json").write_text(
        json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    counts = Counter(tok for text in texts for tok in text.split())
    by_type = found[2:].mean()
    by_token = sum(counts[w] for w, f in zip(vocab, found, strict=True) if f) / sum(counts.values())
    print(f"vocab={len(vocab)} dim={matrix.shape[1]} -> {stem}.npy/.json")
    print(f"fastText coverage: {by_type:.2%} of word types, {by_token:.2%} of Train tokens")


if __name__ == "__main__":
    main()

import numpy as np
import torch
from torch import nn

from absa.data import ASPECTS
from absa.training import fit_loop, set_seed

N = len(ASPECTS)


def test_set_seed_reproducible() -> None:
    set_seed(1)
    a = torch.rand(3)
    set_seed(1)
    assert torch.equal(a, torch.rand(3))


def _batches():
    return [{"x": torch.ones(4, 1), "y": torch.zeros(4, 1)}]


def _loss(module: nn.Module, batch: dict) -> torch.Tensor:
    return ((module(batch["x"]) - batch["y"]) ** 2).mean()


def test_early_stopping_restores_best() -> None:
    torch.manual_seed(0)
    module = nn.Linear(1, 1)
    Y_dev = np.zeros((2, N), dtype=np.int8)
    Y_dev[:, 0] = 1
    wrong = np.zeros_like(Y_dev)
    wrong[:, 0] = 3
    outputs = iter([Y_dev, wrong, wrong, wrong])  # sentiment macro-F1: 1.0, 0.0, 0.0, 0.0
    saved = []

    def predict_dev():
        saved.append(module.weight.detach().clone())
        return next(outputs)

    result = fit_loop(
        module,
        _batches,
        predict_dev,
        Y_dev,
        _loss,
        epochs=5,
        lr=0.1,
        patience=2,
        log=lambda m: None,
    )
    assert result["best_epoch"] == 1
    assert result["best_dev_sentiment_macro_f1"] == 1.0
    assert len(result["history"]) == 3
    assert torch.equal(module.weight.detach(), saved[0])
    assert not torch.equal(saved[0], saved[2])


def test_no_dev_runs_all_epochs() -> None:
    result = fit_loop(
        nn.Linear(1, 1), _batches, None, None, _loss, epochs=2, lr=0.1, log=lambda m: None
    )
    assert len(result["history"]) == 2
    assert result["best_dev_sentiment_macro_f1"] is None

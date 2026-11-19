"""Seeding and a generic training loop with early stopping on Dev sentiment macro-F1."""

import random
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from transformers import get_linear_schedule_with_warmup

from absa.evaluate import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_loop(
    module: nn.Module,
    train_batches: Callable[[], Iterable[dict]],
    predict_dev: Callable[[], np.ndarray] | None,
    Y_dev: np.ndarray | None,
    loss_fn: Callable[[nn.Module, dict], Tensor],
    *,
    epochs: int,
    lr: float,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    patience: int = 2,
    fp16: bool = False,
    device: str = "cpu",
    grad_clip: float = 1.0,
    log: Callable[[str], None] = print,
) -> dict:
    module.to(device)
    total_steps = sum(1 for _ in train_batches()) * epochs
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), max(total_steps, 1)
    )
    use_amp = fp16 and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_state, best_epoch, bad_epochs = None, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        module.train()
        loss_sum, n_batches = 0.0, 0
        for batch in train_batches():
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(module, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()
            n_batches += 1

        record = {"epoch": epoch, "train_loss": loss_sum / max(n_batches, 1)}
        if predict_dev is None:
            best_epoch = epoch
        else:
            score = evaluate(Y_dev, predict_dev())["sentiment_macro_f1"]
            record["dev_sentiment_macro_f1"] = score
            if best_score is None or score > best_score:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            else:
                bad_epochs += 1
        history.append(record)
        log(
            " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
            )
        )
        if predict_dev is not None and bad_epochs >= patience:
            break

    if best_state is not None:
        module.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_dev_sentiment_macro_f1": best_score,
        "history": history,
    }

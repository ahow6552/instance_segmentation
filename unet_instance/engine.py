"""訓練 / 評估迴圈"""
import math
import time

import torch
import torch.nn.functional as F

from .coco_utils import coco_eval
from .models import encoders, forward_all, split_outputs
from .postprocess import encode_instances, postprocess


def autocast(cfg, device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg["amp"] and device.type == "cuda")


def to_device(targets, device):
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def build_optimizer_and_scheduler(model, cfg, iters_per_epoch):
    encoder_params = [p for enc in encoders(model) for p in enc.parameters()]
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["lr_encoder"]},
        {"params": other_params, "lr": cfg["lr_decoder"]},
    ], weight_decay=cfg["weight_decay"])

    total = cfg["num_epochs"] * iters_per_epoch
    warmup = cfg["warmup_iters"]

    def lr_lambda(it):   # 以 iteration 為單位：線性 warmup，之後 cosine 衰減
        if it < warmup:
            return 0.001 + (1 - 0.001) * it / warmup
        progress = (it - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, logger=None, epoch=0):
    model.train()
    total_loss, sums, n = 0.0, {}, 0
    t0 = time.time()

    for it, (images, targets, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = to_device(targets, device)

        with autocast(cfg, device):
            outs = forward_all(model, images)
        loss, parts = criterion(outs, targets)

        if not torch.isfinite(loss):
            if logger:
                logger.warning(f"non-finite loss, skip batch: {parts}")
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1

        if logger and cfg["log_every"] and it % cfg["log_every"] == 0:
            speed = (time.time() - t0) / it
            logger.info(f"  epoch {epoch} iter {it}/{len(loader)} loss {total_loss / n:.4f} "
                        f"lr {optimizer.param_groups[1]['lr']:.2e} ({speed:.2f}s/it, "
                        f"eta {speed * (len(loader) - it) / 60:.1f} min)")

    n = max(n, 1)
    return total_loss / n, {k: v / n for k, v in sums.items()}


@torch.no_grad()
def predict(model, images, cfg, device, num_classes, stage=-1):
    """一個 batch → (每張圖的 [masks, labels, scores] list, 各階段原始輸出)"""
    with autocast(cfg, device):
        outs = forward_all(model, images)
    out = split_outputs(outs[stage].float(), num_classes, cfg["offset_scale"])
    sem_prob = F.softmax(out["semantic"], dim=1)
    preds = [postprocess(sem_prob[b], out["center"][b, 0], out["offset"][b], cfg) for b in range(images.shape[0])]
    return preds, out, outs


@torch.no_grad()
def evaluate(model, loader, cfg, device, num_classes, criterion=None, stage=-1):
    """回傳 (val loss 或 None, COCO 結果 list)；stage=-1 用最後一階段，0 用第一階段"""
    model.eval()
    total_loss, n = 0.0, 0
    results = []

    for images, targets, meta in loader:
        images = images.to(device, non_blocking=True)
        preds, _, outs = predict(model, images, cfg, device, num_classes, stage)

        if criterion is not None:
            loss, _ = criterion.single(outs[stage], to_device(targets, device))
            total_loss += loss.item()
            n += 1

        for b, (masks, labels, scores) in enumerate(preds):
            results += encode_instances(masks, labels, scores, int(meta["image_id"][b]),
                                        int(meta["orig_h"][b]), int(meta["orig_w"][b]))

    return (total_loss / max(n, 1) if criterion is not None else None), results


def evaluate_map(model, loader, cfg, device, num_classes, coco_gt, img_ids, criterion=None, stage=-1):
    """evaluate + COCO segm 評估，回傳 (loss, COCOeval 或 None)"""
    loss, results = evaluate(model, loader, cfg, device, num_classes, criterion, stage)
    return loss, coco_eval(coco_gt, results, img_ids, "segm")

"""
訓練 U-Net instance segmentation（單階段或 stacked）。

    python train.py --config configs/stacked.yaml --gpu 0
    python train.py --config configs/stacked.yaml --gpu 0 --set batch_size=6 num_epochs=60
    python train.py --resume Result/20261027_101500_unet_instance_stacked --gpu 0    # 中斷後接著訓練
    python train.py --config configs/stacked.yaml --gpu 0 --sanity                  # 只做單一 batch 過擬合測試

輸出（Result/<時間>_<run_name>/）：
    config.yaml  train.log  history.json  curves.png  best.pth（val mask mAP 最高）  last.pth（可 resume）
    val_predictions.png  test_predictions.png
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/stacked.yaml", help="YAML 設定檔")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="覆蓋設定，例如 batch_size=4")
    p.add_argument("--gpu", default=None, help='設定 CUDA_VISIBLE_DEVICES，例如 0；給 "" 代表用 CPU')
    p.add_argument("--resume", default=None, help="Result/<run> 資料夾，從裡面的 last.pth 接著訓練")
    p.add_argument("--sanity", action="store_true", help="只做單一 batch 過擬合測試，不建立 run 資料夾")
    p.add_argument("--skip-test", action="store_true", help="訓練完不跑測試集評估")
    return p.parse_args()


def main():
    args = parse_args()
    # 路徑先轉成絕對路徑，再切到專案資料夾（從哪裡啟動都能找到 Dataset/、Result/）
    config_path = os.path.abspath(args.config) if args.config else None
    resume_dir = os.path.abspath(args.resume) if args.resume else None
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu      # 必須在 import torch 之前設定
    os.chdir(HERE)
    sys.path.insert(0, HERE)

    import torch
    from unet_instance.config import load_config, save_config
    from unet_instance.data import DataBundle
    from unet_instance.engine import build_optimizer_and_scheduler, evaluate_map, train_one_epoch, autocast, to_device
    from unet_instance.losses import MultiStageLoss
    from unet_instance.models import build_model, forward_all
    from unet_instance.report import load_gts, split_report, stage_comparison
    from unet_instance.utils import (get_device, load_json, load_weights, new_run_dir, save_checkpoint, save_json,
                                     set_seed, setup_logger)
    from unet_instance.viz import plot_curves, save_gt_vs_pred

    # ---------------------------------------------------------------- config / run dir
    if resume_dir:
        cfg = load_config(os.path.join(resume_dir, "config.yaml"), args.set)
        run_dir = resume_dir
    else:
        cfg = load_config(config_path, args.set)
        run_dir = None if args.sanity else new_run_dir(cfg["results_dir"], cfg["run_name"])

    logger = setup_logger(run_dir)
    device = get_device()
    set_seed(cfg["seed"])
    logger.info(f"device: {device}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    if run_dir:
        logger.info(f"run dir: {run_dir}")
        if not resume_dir:
            save_config(cfg, os.path.join(run_dir, "config.yaml"))

    bundle = DataBundle(cfg, logger, build_train=not args.sanity)
    model = build_model(cfg, bundle.num_classes, logger).to(device)
    criterion = MultiStageLoss(bundle.num_classes, cfg)

    # ---------------------------------------------------------------- sanity check
    if args.sanity:
        from torch.utils.data import DataLoader
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")
        images, targets, _ = next(iter(DataLoader(bundle.val_ds, batch_size=2)))
        images, targets = images.to(device), to_device(targets, device)
        model.train()
        for step in range(60):
            with autocast(cfg, device):
                outs = forward_all(model, images)
            loss, parts = criterion(outs, targets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % 10 == 0 or step == 59:
                logger.info(f"sanity step {step:2d}  loss {loss.item():.4f}  " +
                            "  ".join(f"{k} {v:.4f}" for k, v in parts.items()))
        return

    # ---------------------------------------------------------------- optimizer / resume
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, len(bundle.train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")
    gts = load_gts(cfg)

    history = {"epoch": [], "train_loss": [], "train_parts": [], "val_epoch": [],
               "val_loss": [], "val_segm_map": [], "val_segm_map50": [], "lr": []}
    best_map, best_epoch, start_epoch = -1.0, -1, 0
    best_path, last_path = os.path.join(run_dir, "best.pth"), os.path.join(run_dir, "last.pth")

    if resume_dir:
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history, best_map, best_epoch, start_epoch = ckpt["history"], ckpt["best_map"], ckpt["best_epoch"], ckpt["epoch"]
        logger.info(f"resumed from {last_path}: 已完成 {start_epoch} 個 epoch，best val mAP {best_map:.4f} @ {best_epoch}")

    # ---------------------------------------------------------------- training loop
    for epoch in range(start_epoch, cfg["num_epochs"]):
        t0 = time.time()
        lr_now = optimizer.param_groups[1]["lr"]
        train_loss, parts = train_one_epoch(model, bundle.train_loader, criterion, optimizer, scheduler, scaler,
                                            cfg, device, logger, epoch + 1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_parts"].append(parts)
        history["lr"].append(lr_now)

        stage_txt = "  ".join(f"{k} {v:.3f}" for k, v in parts.items() if k.endswith("_total"))
        msg = (f"Epoch [{epoch + 1}/{cfg['num_epochs']}] lr {lr_now:.2e} | train loss {train_loss:.4f} "
               f"(sem {parts['sem']:.3f} ctr {parts['center']:.5f} off {parts['offset']:.2f} | {stage_txt})")

        if (epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["num_epochs"]:
            val_loss, E = evaluate_map(model, bundle.val_loader, cfg, device, bundle.num_classes,
                                       gts["train"], bundle.val_img_ids, criterion)
            segm_map = float(E.stats[0]) if E is not None else 0.0
            segm_map50 = float(E.stats[1]) if E is not None else 0.0
            history["val_epoch"].append(epoch + 1)
            history["val_loss"].append(val_loss)
            history["val_segm_map"].append(segm_map)
            history["val_segm_map50"].append(segm_map50)
            msg += f" | val loss {val_loss:.4f} | val mask mAP {segm_map:.4f} (AP50 {segm_map50:.4f})"

            if segm_map > best_map:
                best_map, best_epoch = segm_map, epoch + 1
                torch.save(model.state_dict(), best_path)
                msg += "  -> saved best"
            plot_curves(history, cfg, os.path.join(run_dir, "curves.png"))

        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch + 1, history, best_map, best_epoch)
        save_json({"history": history, "config": cfg, "best_epoch": best_epoch, "best_val_segm_map": best_map},
                  os.path.join(run_dir, "history.json"))
        logger.info(msg + f"  [{time.time() - t0:.0f}s]")

    logger.info(f"best val mask mAP {best_map:.4f} @ epoch {best_epoch}")

    # ---------------------------------------------------------------- test / report
    if args.skip_test:
        return
    if not os.path.exists(best_path):
        logger.warning("找不到 best.pth（沒有任何一次 val 評估），略過測試")
        return
    load_weights(model, best_path, device)
    payload = load_json(os.path.join(run_dir, "history.json"))
    payload.update(split_report(model, bundle, gts, "test", cfg, device, logger))
    payload["stage_comparison"] = stage_comparison(model, bundle, gts, cfg, device, logger)
    save_json(payload, os.path.join(run_dir, "history.json"))

    save_gt_vs_pred(model, bundle.eval_full, bundle.val_idx, cfg, device, bundle.num_classes, bundle.id_to_name,
                    os.path.join(run_dir, "val_predictions.png"))
    save_gt_vs_pred(model, bundle.test_ds, list(range(len(bundle.test_ds))), cfg, device, bundle.num_classes,
                    bundle.id_to_name, os.path.join(run_dir, "test_predictions.png"))
    logger.info(f"done. 結果在 {run_dir}")


if __name__ == "__main__":
    main()

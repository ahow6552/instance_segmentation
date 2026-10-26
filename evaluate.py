"""
評估訓練好的模型（不用重新訓練）。

    # 評估某次訓練的 best.pth（設定從該 run 的 config.yaml 讀）
    python evaluate.py --run-dir Result/20261027_101500_unet_instance_stacked --gpu 0
    python evaluate.py --run-dir Result/... --split val test --stages --viz 4
    python evaluate.py --run-dir Result/... --tune                        # 在 val 上調 center_thr / nms_kernel
    python evaluate.py --run-dir Result/... --set center_thr=0.2 nms_kernel=5

    # 評估 notebook 訓練出來的單階段權重
    python evaluate.py --config configs/single.yaml --ckpt best_unet_instance.pth --gpu 0

結果存成 <run-dir 或 --out>/eval_<時間>.json，log 存成 eval_<時間>.log。
"""
import argparse
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="Result/<run>：讀它的 config.yaml，權重預設用 best.pth")
    p.add_argument("--config", default=None, help="沒有 run-dir 時用的 YAML 設定檔")
    p.add_argument("--ckpt", default=None, help="權重檔（純 state_dict 或 last.pth 都可以）")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="覆蓋設定（例如 post-processing 參數）")
    p.add_argument("--split", nargs="+", default=["test"], choices=["val", "test"])
    p.add_argument("--stages", action="store_true", help="stacked 模型：比較每一階段輸出的 mAP")
    p.add_argument("--tune", action="store_true", help="在 val 上掃 center_thr × nms_kernel")
    p.add_argument("--viz", type=int, default=0, help="每個 split 畫幾張 GT vs 預測（0 = 不畫）")
    p.add_argument("--out", default=None, help="輸出資料夾（預設 = run-dir，否則 Result/<時間>_eval）")
    p.add_argument("--gpu", default=None, help='設定 CUDA_VISIBLE_DEVICES；給 "" 代表用 CPU')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.run_dir and not (args.config and args.ckpt):
        sys.exit("請給 --run-dir，或同時給 --config 和 --ckpt")
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else None
    config_path = os.path.join(run_dir, "config.yaml") if run_dir else os.path.abspath(args.config)
    ckpt = os.path.abspath(args.ckpt) if args.ckpt else os.path.join(run_dir, "best.pth")
    out_dir = os.path.abspath(args.out) if args.out else run_dir
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.chdir(HERE)
    sys.path.insert(0, HERE)

    import logging
    from unet_instance.config import load_config
    from unet_instance.data import DataBundle
    from unet_instance.models import build_model
    from unet_instance.report import load_gts, split_report, stage_comparison, tune_postprocess
    from unet_instance.utils import get_device, load_weights, new_run_dir, save_json, setup_logger
    from unet_instance.viz import save_gt_vs_pred

    cfg = load_config(config_path, args.set)
    if out_dir is None:
        out_dir = new_run_dir(cfg["results_dir"], "eval")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = setup_logger(None, name="unet_instance_eval")
    fh = logging.FileHandler(os.path.join(out_dir, f"eval_{stamp}.log"), encoding="utf-8")
    fh.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(fh)

    device = get_device()
    logger.info(f"device: {device}  config: {config_path}  ckpt: {ckpt}")
    cfg["encoder_weights"] = None              # 權重會從 ckpt 載入，不用再下載預訓練權重
    cfg["stage2_encoder_weights"] = None
    cfg["stage1_init"] = None

    bundle = DataBundle(cfg, logger, build_train=False)
    model = build_model(cfg, bundle.num_classes, logger).to(device)
    load_weights(model, ckpt, device)
    model.eval()
    gts = load_gts(cfg)

    payload = {"config": cfg, "ckpt": ckpt, "overrides": args.set}
    if args.tune:
        payload["tune"], best = tune_postprocess(model, bundle, gts, cfg, device, logger)
        logger.info(f"之後可以用 --set center_thr={best['center_thr']} nms_kernel={best['nms_kernel']}")
    for split in args.split:
        payload.update(split_report(model, bundle, gts, split, cfg, device, logger))
    if args.stages:
        payload["stage_comparison"] = stage_comparison(model, bundle, gts, cfg, device, logger, splits=args.split)
    if args.viz:
        for split in args.split:
            ds, idx = ((bundle.eval_full, bundle.val_idx) if split == "val"
                       else (bundle.test_ds, list(range(len(bundle.test_ds)))))
            path = os.path.join(out_dir, f"eval_{stamp}_{split}_predictions.png")
            save_gt_vs_pred(model, ds, idx, cfg, device, bundle.num_classes, bundle.id_to_name, path, n=args.viz)
            logger.info(f"saved {path}")

    save_json(payload, os.path.join(out_dir, f"eval_{stamp}.json"))
    logger.info(f"saved {os.path.join(out_dir, f'eval_{stamp}.json')}")


if __name__ == "__main__":
    main()

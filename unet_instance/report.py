"""train.py / evaluate.py 共用的評估流程：split 評估、測試報告、階段比較、post-processing 調參"""
from .coco_utils import coco_eval, load_coco, per_class_ap, stats_dict
from .engine import evaluate
from .models import num_stages


def load_gts(cfg):
    # val 是從 train json 切出來的，GT 要用 train json
    return {"train": load_coco(cfg["train_json"]), "test": load_coco(cfg["test_json"])}


def split_spec(bundle, gts, split):
    if split == "val":
        return bundle.val_loader, gts["train"], bundle.val_img_ids
    if split == "test":
        return bundle.test_loader, gts["test"], bundle.test_img_ids
    raise ValueError(f"split 必須是 val 或 test，收到 {split!r}")


def eval_split(model, bundle, gts, split, cfg, device, stage=-1):
    """回傳 (COCOeval 或 None, summary 文字)"""
    loader, gt, ids = split_spec(bundle, gts, split)
    _, results = evaluate(model, loader, cfg, device, bundle.num_classes, stage=stage)
    return coco_eval(gt, results, ids, "segm", verbose=True)


def split_report(model, bundle, gts, split, cfg, device, logger):
    E, summary = eval_split(model, bundle, gts, split, cfg, device)
    logger.info(f"===== {split}: segm =====\n{summary}")
    if E is None:
        return {f"{split}_segm": None, f"{split}_per_class_mask_ap": None}
    class_ap = per_class_ap(E, bundle.id_to_name)
    logger.info(f"===== {split}: per-class mask AP =====\n" + "\n".join(
        f"  {name:>12s}: {ap:.4f}" for name, ap in sorted(class_ap.items(), key=lambda x: -x[1] if x[1] == x[1] else 1)))
    return {f"{split}_segm": stats_dict(E), f"{split}_per_class_mask_ap": class_ap}


def stage_comparison(model, bundle, gts, cfg, device, logger, splits=("val", "test")):
    """stacked 模型：分別用每一階段的輸出評估；單階段模型直接回傳 None"""
    if num_stages(model) < 2:
        return None
    out = {}
    for split in splits:
        for s in range(num_stages(model)):
            E, _ = eval_split(model, bundle, gts, split, cfg, device, stage=s)
            out[f"{split}_stage{s + 1}"] = stats_dict(E)
            if E is not None:
                st = E.stats
                logger.info(f"{split:>4s} stage {s + 1}: mask AP {st[0]:.4f}  AP50 {st[1]:.4f}  AP75 {st[2]:.4f}  "
                            f"AP_small {st[3]:.4f}  AR100 {st[8]:.4f}")
    return out


def tune_postprocess(model, bundle, gts, cfg, device, logger, thrs=(0.05, 0.1, 0.2), kernels=(5, 7, 11)):
    """在 val 上掃 center_thr × nms_kernel，回傳 (所有結果, 最佳設定)"""
    rows, best = [], None
    for thr in thrs:
        for k in kernels:
            E, _ = eval_split(model, bundle, gts, "val", {**cfg, "center_thr": thr, "nms_kernel": k}, device)
            ap = float(E.stats[0]) if E is not None else 0.0
            rows.append({"center_thr": thr, "nms_kernel": k, "val_mask_ap": ap})
            logger.info(f"center_thr {thr:<5} nms_kernel {k:<3} -> val mask AP {ap:.4f}")
            if best is None or ap > best["val_mask_ap"]:
                best = rows[-1]
    logger.info(f"best: center_thr {best['center_thr']}  nms_kernel {best['nms_kernel']}  "
                f"val mask AP {best['val_mask_ap']:.4f}")
    return rows, best

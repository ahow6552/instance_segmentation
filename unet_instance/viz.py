"""畫圖（只存檔、不顯示，tmux / 無螢幕環境可用）"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .engine import predict


def plot_curves(history, cfg, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["val_epoch"], history["val_loss"], "o-", label="val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Total loss"); axes[0].legend()

    for k, w in [("sem", cfg["w_sem"]), ("center", cfg["w_center"]), ("offset", cfg["w_offset"])]:
        axes[1].plot(history["epoch"], [p[k] * w for p in history["train_parts"]], label=f"{k} (x{w:g})")
    if history["train_parts"] and "stage2_total" in history["train_parts"][0]:
        axes[1].plot(history["epoch"], [p["stage1_total"] for p in history["train_parts"]], "--",
                     label="stage1 total")
    axes[1].set_xlabel("Epoch"); axes[1].set_title("Train loss parts (last stage, weighted)"); axes[1].legend()

    axes[2].plot(history["val_epoch"], history["val_segm_map"], "o-", label="mask mAP@[.5:.95]")
    axes[2].plot(history["val_epoch"], history["val_segm_map50"], "o-", label="mask AP50")
    axes[2].set_xlabel("Epoch"); axes[2].set_title("Validation mAP"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_instances(ax, image, masks, labels, id_to_name, scores=None, title=""):
    """image: (H, W, 3) float；masks: (K, H, W) bool"""
    ax.imshow(image)
    cmap = plt.get_cmap("tab20")
    overlay = np.zeros((*image.shape[:2], 4))
    for k, (m, lab) in enumerate(zip(masks, labels)):
        color = cmap(k % 20)
        overlay[m] = (*color[:3], 0.5)
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        text = id_to_name.get(int(lab), str(int(lab)))
        if scores is not None:
            text += f" {float(scores[k]):.2f}"
        ax.text(xs.min(), max(ys.min() - 3, 0), text, color=color, fontsize=8, weight="bold")
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")


@torch.no_grad()
def save_gt_vs_pred(model, dataset_full, indices, cfg, device, num_classes, id_to_name, path,
                    n=3, score_threshold=0.3, seed=0):
    model.eval()
    n = min(n, len(indices))
    idxs = np.random.default_rng(seed).choice(indices, n, replace=False)
    fig, axes = plt.subplots(n, 4, figsize=(24, 4.5 * n))
    axes = np.atleast_2d(axes)

    for row, i in zip(axes, idxs):
        img, inst_map, gt_labels, _ = dataset_full.load(i)
        img_np = np.array(img) / 255.0
        if len(gt_labels):
            gt_masks = np.stack([inst_map == k for k in range(1, len(gt_labels) + 1)])
        else:
            gt_masks = np.zeros((0, *inst_map.shape), bool)
        draw_instances(row[0], img_np, gt_masks, gt_labels, id_to_name, title=f"GT: {len(gt_labels)} object(s)")

        image, _, _ = dataset_full[i]
        preds, out, _ = predict(model, image[None].to(device), cfg, device, num_classes)
        masks, labels, scores = preds[0]
        keep = scores > score_threshold
        draw_instances(row[1], img_np, masks[keep.to(masks.device)].cpu().numpy(), labels[keep], id_to_name,
                       scores[keep], title=f"Pred: {int(keep.sum())} object(s)")

        row[2].imshow(img_np)
        row[2].imshow(out["center"][0, 0].cpu().clamp(0, 1), cmap="hot", alpha=0.6)
        row[2].set_title("pred center heatmap"); row[2].axis("off")

        sem_pred = F.softmax(out["semantic"], dim=1)[0].argmax(0).cpu()
        row[3].imshow(sem_pred, vmin=0, vmax=num_classes - 1, cmap="tab20")
        row[3].set_title("pred semantic"); row[3].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

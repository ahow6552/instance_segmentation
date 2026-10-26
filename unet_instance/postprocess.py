"""Post-processing：center NMS → 像素依 offset 投票給最近的中心 → 多數決類別 → COCO RLE"""
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils


@torch.no_grad()
def find_centers(center, thr, nms_kernel, top_k):
    """center: (H, W) → 峰值座標 (K, 2)[y, x] 和分數 (K,)"""
    pooled = F.max_pool2d(center[None, None], nms_kernel, stride=1, padding=nms_kernel // 2)[0, 0]
    keep = (center == pooled) & (center > thr)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = center[ys, xs]
    if len(scores) > top_k:
        idx = scores.topk(top_k).indices
        ys, xs, scores = ys[idx], xs[idx], scores[idx]
    return torch.stack([ys, xs], dim=1).float(), scores


@torch.no_grad()
def postprocess(sem_prob, center, offset, cfg):
    """
    sem_prob: (C, H, W) softmax 後；center: (H, W)；offset: (2, H, W)（pixel）
    回傳 masks (K, H, W) bool、labels (K,)、scores (K,)
    """
    C, H, W = sem_prob.shape
    empty = (torch.zeros((0, H, W), dtype=torch.bool, device=sem_prob.device),
             torch.zeros((0,), dtype=torch.long), torch.zeros((0,)))

    sem_pred = sem_prob.argmax(0)
    fg = sem_pred > 0
    centers, center_scores = find_centers(center, cfg["center_thr"], cfg["nms_kernel"], cfg["top_k"])
    if len(centers) == 0 or not fg.any():
        return empty

    # 1) 每個前景像素：位置 + offset → 最近的中心
    ys, xs = torch.nonzero(fg, as_tuple=True)
    voted = torch.stack([ys + offset[0, ys, xs], xs + offset[1, ys, xs]], dim=1)
    assign = torch.cdist(voted, centers).argmin(dim=1)
    K = len(centers)

    # 2) 類別 = semantic 多數決（只在前景類別中投票）
    pix_cls = sem_pred[ys, xs]
    votes = torch.bincount(assign * C + pix_cls, minlength=K * C).view(K, C)
    votes[:, 0] = 0
    inst_cls = votes.argmax(dim=1)
    area = votes.sum(dim=1)

    # 3) 分數 = 中心 heatmap 值 × 物件內「該類別」的平均機率
    pix_prob = sem_prob[inst_cls[assign], ys, xs]
    mean_prob = torch.bincount(assign, weights=pix_prob, minlength=K) / area.clamp(min=1)
    inst_scores = center_scores.clamp(max=1) * mean_prob

    valid = area >= cfg["min_area"]
    if not valid.any():
        return empty
    order = inst_scores[valid].argsort(descending=True)[:cfg["max_dets"]]
    keep_ids = torch.nonzero(valid).squeeze(1)[order]

    inst_map = torch.full((H, W), -1, dtype=torch.long, device=sem_prob.device)
    inst_map[ys, xs] = assign
    masks = inst_map[None] == keep_ids[:, None, None]
    return masks, inst_cls[keep_ids].cpu(), inst_scores[keep_ids].cpu()


def encode_instances(masks, labels, scores, img_id, orig_h, orig_w):
    """input_size 的 mask → 原圖大小 → COCO RLE 結果 list"""
    results = []
    if len(masks) == 0:
        return results
    masks = F.interpolate(masks[None].float(), size=(orig_h, orig_w), mode="bilinear",
                          align_corners=False)[0] > 0.5
    masks = masks.cpu().numpy().astype(np.uint8)
    for m, lab, sc in zip(masks, labels.tolist(), scores.tolist()):
        if m.sum() == 0:
            continue
        rle = maskUtils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("ascii")
        x, y, w, h = maskUtils.toBbox(rle).tolist()
        results.append({"image_id": img_id, "category_id": int(lab), "score": float(sc),
                        "segmentation": rle, "bbox": [x, y, w, h]})
    return results

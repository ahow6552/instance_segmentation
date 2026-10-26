"""Loss：semantic CE + Dice、center MSE、offset L1；stacked 時每個階段加權相加"""
import torch.nn as nn
import torch.nn.functional as F

from .models import split_outputs


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=1.0, eps=1e-6, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, target):
        ce = self.ce(logits, target)

        prob = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = (prob * target_onehot).sum(dims)
        union = prob.sum(dims) + target_onehot.sum(dims)
        dice_per_class = (2 * inter + self.eps) / (union + self.eps)

        present = target_onehot.sum(dims) > 0
        if present.any():
            dice_loss = 1 - dice_per_class[present].mean()
        else:
            dice_loss = 1 - dice_per_class.mean()

        return ce + self.dice_weight * dice_loss


class InstanceLoss(nn.Module):
    """單一階段輸出的 loss"""

    def __init__(self, num_classes, cfg):
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg
        self.sem_loss = CEDiceLoss(num_classes)

    def forward(self, output, targets):
        out = split_outputs(output.float(), self.num_classes, self.cfg["offset_scale"])   # AMP 下一律用 float32 算

        sem = self.sem_loss(out["semantic"], targets["semantic"])
        center = F.mse_loss(out["center"], targets["center"])

        w = targets["offset_weight"]
        offset = (F.l1_loss(out["offset"], targets["offset"], reduction="none") * w).sum() / (w.sum() * 2 + 1)

        total = self.cfg["w_sem"] * sem + self.cfg["w_center"] * center + self.cfg["w_offset"] * offset
        return total, {"sem": sem.item(), "center": center.item(), "offset": offset.item()}


class MultiStageLoss(nn.Module):
    """outs = [out1, out2, ...]；最後一階段權重 1，前面的階段用 stage1_loss_weight（intermediate supervision）"""

    def __init__(self, num_classes, cfg):
        super().__init__()
        self.single = InstanceLoss(num_classes, cfg)
        self.prev_weight = cfg["stage1_loss_weight"]

    def forward(self, outs, targets):
        total = 0.0
        parts = {}
        for s, out in enumerate(outs):
            loss, p = self.single(out, targets)
            last = s == len(outs) - 1
            total = total + (1.0 if last else self.prev_weight) * loss
            parts[f"stage{s + 1}_total"] = loss.item()
            if last:
                parts.update(p)        # sem / center / offset 記錄最後一階段的
        return total, parts

"""資料：COCO polygon → instance id map → semantic / center / offset targets；train / val / test 切分與 DataLoader"""
import json
import math
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools import mask as maskUtils
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def make_targets(inst_map, inst_labels, sigma):
    """inst_map: (H, W) int，0 = 背景、k = 第 k 個物件；inst_labels[k-1] 是它的類別"""
    H, W = inst_map.shape
    semantic = np.zeros((H, W), dtype=np.int64)
    center = np.zeros((H, W), dtype=np.float32)
    offset = np.zeros((2, H, W), dtype=np.float32)
    offset_weight = np.zeros((H, W), dtype=np.float32)

    r = int(3 * sigma)
    g = np.arange(-r, r + 1, dtype=np.float32)
    gaussian = np.exp(-(g[:, None] ** 2 + g[None, :] ** 2) / (2 * sigma ** 2))

    for k, label in enumerate(inst_labels, start=1):
        ys, xs = np.nonzero(inst_map == k)
        if len(ys) == 0:          # resize 後消失的小物件
            continue
        semantic[ys, xs] = label

        cy, cx = ys.mean(), xs.mean()          # 質心當中心
        offset[0, ys, xs] = cy - ys
        offset[1, ys, xs] = cx - xs
        offset_weight[ys, xs] = 1.0

        # 只在中心附近 (2r+1)^2 的窗口貼 Gaussian
        icy, icx = int(round(cy)), int(round(cx))
        y0, y1 = max(0, icy - r), min(H, icy + r + 1)
        x0, x1 = max(0, icx - r), min(W, icx + r + 1)
        patch = gaussian[y0 - (icy - r):y1 - (icy - r), x0 - (icx - r):x1 - (icx - r)]
        center[y0:y1, x0:x1] = np.maximum(center[y0:y1, x0:x1], patch)

    return {
        "semantic": torch.from_numpy(semantic),
        "center": torch.from_numpy(center)[None],
        "offset": torch.from_numpy(offset),
        "offset_weight": torch.from_numpy(offset_weight)[None],
    }


class UIIS10KUNetInstanceDataset(Dataset):
    def __init__(self, coco, image_dir, input_size, sigma, train=False, keep_empty=False):
        self.image_dir = image_dir
        self.input_size = input_size
        self.sigma = sigma
        self.train = train
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

        self.anns_by_img = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        if keep_empty:
            self.images = list(coco["images"])
        else:
            self.images = [img for img in coco["images"] if img["id"] in self.anns_by_img]

    def __len__(self):
        return len(self.images)

    def load(self, idx):
        """回傳 resize 後的 PIL image、instance map、每個 instance 的類別（不做增強）"""
        info = self.images[idx]
        img = Image.open(os.path.join(self.image_dir, info["file_name"])).convert("RGB")
        width, height = img.size

        anns = sorted(self.anns_by_img.get(info["id"], []), key=lambda a: -a["area"])
        inst_map = np.zeros((height, width), dtype=np.int32)
        labels = []
        for ann in anns:
            rles = maskUtils.frPyObjects(ann["segmentation"], height, width)
            m = maskUtils.decode(rles)
            if m.ndim == 3:
                m = m.max(axis=2)
            labels.append(ann["category_id"])
            inst_map[m > 0] = len(labels)

        img = img.resize(self.input_size, Image.BILINEAR)
        inst_map = np.array(Image.fromarray(inst_map).resize(self.input_size, Image.NEAREST))
        return img, inst_map, np.array(labels, dtype=np.int64), info

    def __getitem__(self, idx):
        img, inst_map, labels, info = self.load(idx)

        if self.train:
            img = self.color_jitter(img)
            # 水平翻轉（用 torch 的 RNG，DataLoader 的每個 worker 會自動拿到不同 seed）
            if torch.rand(1).item() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                inst_map = inst_map[:, ::-1].copy()

        targets = make_targets(inst_map, labels, self.sigma)

        image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        meta = {"image_id": info["id"], "orig_h": info["height"], "orig_w": info["width"]}
        return image, targets, meta


def denormalize(image):
    return (image.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def repeat_factors(dataset_full, indices, t):
    """LVIS 的 Repeat Factor Sampling：rf_c = max(1, sqrt(t / f_c))，每張圖取它包含類別的最大值"""
    img_cats = []
    for i in indices:
        anns = dataset_full.anns_by_img.get(dataset_full.images[i]["id"], [])
        img_cats.append({a["category_id"] for a in anns})

    n = len(img_cats)
    freq = {}
    for cats in img_cats:
        for c in cats:
            freq[c] = freq.get(c, 0) + 1
    cat_rf = {c: max(1.0, math.sqrt(t / (k / n))) for c, k in freq.items()}
    img_rf = [max((cat_rf[c] for c in cats), default=1.0) for cats in img_cats]
    return img_rf, cat_rf


class DataBundle:
    """一次建好 train / val / test 的 dataset、loader 和 metadata，train.py / evaluate.py 共用"""

    def __init__(self, cfg, logger=None, build_train=True):
        log = logger.info if logger else print
        with open(cfg["train_json"]) as f:
            data = json.load(f)
        with open(cfg["test_json"]) as f:
            test_data = json.load(f)

        self.num_classes = len(data["categories"]) + 1   # +1 是背景
        self.id_to_name = {0: "background", **{c["id"]: c["name"] for c in data["categories"]}}

        kw = dict(input_size=cfg["input_size"], sigma=cfg["center_sigma"])
        self.train_full = UIIS10KUNetInstanceDataset(data, cfg["image_dir"], train=True, **kw)
        self.eval_full = UIIS10KUNetInstanceDataset(data, cfg["image_dir"], train=False, **kw)
        self.test_ds = UIIS10KUNetInstanceDataset(test_data, cfg["image_dir"], train=False, keep_empty=True, **kw)

        # 跟 notebook / Mask R-CNN 版相同的切法：只保留有標註的圖、seed 42、10%
        perm = np.random.default_rng(cfg["seed"]).permutation(len(self.train_full))
        n_val = int(len(perm) * cfg["val_ratio"])
        self.val_idx, self.train_idx = perm[:n_val], perm[n_val:]

        if cfg["limit"]:
            n = int(cfg["limit"])
            self.train_idx, self.val_idx = self.train_idx[:n], self.val_idx[:n]
            self.test_ds.images = self.test_ds.images[:n]
            log(f"[limit] 只用 train/val/test 各前 {n} 張（debug 模式）")

        self.train_ds = Subset(self.train_full, self.train_idx)
        self.val_ds = Subset(self.eval_full, self.val_idx)
        self.val_img_ids = [self.eval_full.images[i]["id"] for i in self.val_idx]
        self.test_img_ids = [img["id"] for img in self.test_ds.images]
        log(f"train: {len(self.train_ds)}  val: {len(self.val_ds)}  test: {len(self.test_ds)}")

        loader_kw = dict(num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kw)
        self.test_loader = DataLoader(self.test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kw)

        self.train_loader = None
        if build_train:
            sampler = None
            if cfg["use_rfs"]:
                img_rf, cat_rf = repeat_factors(self.train_full, self.train_idx, cfg["rfs_threshold"])
                log("Repeat Factor Sampling: " + ", ".join(
                    f"{self.id_to_name[c]} {cat_rf[c]:.2f}" for c in sorted(cat_rf)))
                sampler = WeightedRandomSampler(img_rf, num_samples=len(self.train_ds), replacement=True)
            self.train_loader = DataLoader(self.train_ds, batch_size=cfg["batch_size"], shuffle=(sampler is None),
                                           sampler=sampler, drop_last=True, **loader_kw)

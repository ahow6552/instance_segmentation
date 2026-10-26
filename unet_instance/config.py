"""設定：預設值 + YAML 設定檔 + 命令列 --set key=value 覆蓋"""
import copy

import yaml

DEFAULTS = {
    # 實驗
    "model": "stacked",               # "single" | "stacked"
    "run_name": "unet_instance_stacked",
    "results_dir": "Result",
    "seed": 42,
    # 資料
    "image_dir": "Dataset/img",
    "train_json": "Dataset/annotations/multiclass_train.json",
    "test_json": "Dataset/annotations/multiclass_test.json",
    "input_size": (512, 384),         # (W, H)，都要能被 32 整除
    "center_sigma": 6,
    "val_ratio": 0.1,
    "use_rfs": True,                  # Repeat Factor Sampling
    "rfs_threshold": 0.1,
    "num_workers": 4,
    "limit": None,                    # debug 用：train / val / test 各只取前 N 張
    # 模型（stage 1 / 單階段）
    "encoder": "resnet34",
    "encoder_weights": "imagenet",
    "decoder_channels": (256, 128, 64, 64, 64),
    # 模型（stacked 的 stage 2）
    "stage2_encoder": "resnet18",
    "stage2_encoder_weights": "imagenet",
    "stage1_loss_weight": 0.5,        # loss = w × loss(out1) + loss(out2)
    "stage1_init": None,              # 例："best_unet_instance.pth"
    # 訓練
    "batch_size": 8,
    "num_epochs": 40,
    "lr_encoder": 1e-4,
    "lr_decoder": 5e-4,
    "weight_decay": 1e-4,
    "warmup_iters": 500,
    "grad_clip": 10.0,
    "amp": True,
    "eval_every": 2,
    "log_every": 100,                 # 每幾個 iteration 印一次 log
    # loss 權重
    "w_sem": 1.0,
    "w_center": 200.0,
    "w_offset": 0.01,
    "offset_scale": 32.0,
    # post-processing
    "center_thr": 0.1,
    "nms_kernel": 7,
    "top_k": 200,
    "min_area": 16,
    "max_dets": 100,
}

TUPLE_KEYS = ("input_size", "decoder_channels")


def _normalize(cfg):
    for k in TUPLE_KEYS:
        if cfg.get(k) is not None:
            cfg[k] = tuple(cfg[k])
    if cfg["model"] not in ("single", "stacked"):
        raise ValueError(f"model 必須是 single 或 stacked，收到 {cfg['model']!r}")
    return cfg


def _update(cfg, new, source):
    unknown = set(new) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"{source} 有未知的設定：{sorted(unknown)}")
    cfg.update(new)


def load_config(path=None, overrides=()):
    """overrides：["batch_size=4", "input_size=[640,480]", "stage1_init=null"] 這類字串，值用 YAML 語法解析"""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _update(cfg, yaml.safe_load(f) or {}, path)
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 格式是 key=value，收到 {item!r}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = yaml.safe_load(value)
    _update(cfg, parsed, "--set")
    return _normalize(cfg)


def save_config(cfg, path):
    out = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

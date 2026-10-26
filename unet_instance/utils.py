"""雜項：logger、run 資料夾、seed、checkpoint、JSON"""
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import torch


def setup_logger(run_dir=None, name="unet_instance"):
    """同時輸出到 stdout（tmux 看得到）和 run_dir/train.log（session 斷了也留得住）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if run_dir:
        fh = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def new_run_dir(results_dir, run_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"{timestamp}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, history, best_map, best_epoch):
    """完整狀態（可以 --resume 接著訓練）"""
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_map": best_map,
        "best_epoch": best_epoch,
    }, tmp)
    os.replace(tmp, path)   # 先寫暫存檔再改名，寫到一半被中斷也不會弄壞舊的 checkpoint


def load_weights(model, path, device):
    """可以讀純 state_dict（best.pth、notebook 的 .pth）或 save_checkpoint 存的完整 checkpoint"""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state and "optimizer" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model

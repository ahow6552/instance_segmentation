# U-Net Instance Segmentation（.py 版）

把 `unet_instance_segmentation.ipynb`（單階段）和 `unet_instance_stacked.ipynb`（stacked）整理成模組，單階段和 stacked 共用同一套程式，用設定檔切換。

## 檔案結構

```
train.py                 訓練入口（含 resume、sanity check、訓練完自動跑測試集）
evaluate.py              評估入口（不用重新訓練：test / val、階段比較、調 post-processing、畫圖）
configs/
  stacked.yaml           stacked U-Net（ResNet34 + ResNet18）
  single.yaml            單階段 U-Net（比較基準）
unet_instance/
  config.py              預設設定 DEFAULTS、讀 YAML、--set 覆蓋
  data.py                Dataset、make_targets、train/val/test 切分、RFS、DataLoader（DataBundle）
  models.py              單階段 U-Net、StackedUNetInstance、build_model
  losses.py              CE+Dice / MSE / L1、多階段加權
  postprocess.py         center NMS → offset 分組 → COCO RLE
  engine.py              optimizer / scheduler、train_one_epoch、evaluate
  coco_utils.py          COCOeval、每類別 AP
  report.py              測試報告、階段比較、post-processing 調參（train.py / evaluate.py 共用）
  viz.py                 曲線、GT vs 預測圖（只存檔，不需要螢幕）
  utils.py               logger、checkpoint、run 資料夾
```

## 在 tmux 裡訓練

```bash
tmux new -s stacked                      # 開一個 session
cd /ahow/UnetLearning
python train.py --config configs/stacked.yaml --gpu 0
# Ctrl-b 然後按 d：離開 session，訓練繼續在背景跑
```

```bash
tmux attach -t stacked                   # 回來看進度
tail -f Result/<run>/train.log           # 不進 tmux 也可以看 log
tmux ls                                  # 列出所有 session
```

先確認能跑再正式訓練：

```bash
python train.py --config configs/stacked.yaml --gpu 0 --sanity                      # 單一 batch 過擬合測試，loss 要下降
python train.py --config configs/stacked.yaml --gpu 0 --set limit=32 num_epochs=1   # 小資料跑完整流程
```

## 常用選項

```bash
# 覆蓋設定（值用 YAML 語法）
python train.py --config configs/stacked.yaml --gpu 0 --set batch_size=6 num_epochs=60 input_size=[640,480]

# 用單階段權重初始化 stage 1
python train.py --config configs/stacked.yaml --gpu 0 --set stage1_init=best_unet_instance.pth

# tmux 斷掉 / 機器重開：從 last.pth 接著訓練
python train.py --resume Result/<run> --gpu 0

# 單階段比較基準
python train.py --config configs/single.yaml --gpu 1
```

## 評估

```bash
python evaluate.py --run-dir Result/<run> --gpu 0                                    # 測試集
python evaluate.py --run-dir Result/<run> --gpu 0 --split val test --stages --viz 4  # stage1 vs stage2、畫圖
python evaluate.py --run-dir Result/<run> --gpu 0 --tune                             # 在 val 上調 center_thr / nms_kernel
python evaluate.py --run-dir Result/<run> --gpu 0 --set center_thr=0.2 nms_kernel=5  # 用調好的參數評估
python evaluate.py --config configs/single.yaml --ckpt best_unet_instance.pth --gpu 0  # notebook 訓練的權重
```

## 輸出（`Result/<時間>_<run_name>/`）

| 檔案 | 內容 |
|---|---|
| `config.yaml` | 這次用的完整設定 |
| `train.log` | 訓練 log（跟 tmux 畫面相同） |
| `history.json` | 每個 epoch 的 loss / mAP；訓練完會加上 test 指標、每類別 AP、stage 比較 |
| `curves.png` | loss 和 val mAP 曲線（每次 val 評估都會更新，可以訓練中途打開看） |
| `best.pth` | val mask mAP 最高的權重（純 state_dict） |
| `last.pth` | 最後一個 epoch 的完整狀態（含 optimizer），給 `--resume` 用 |
| `val_predictions.png` / `test_predictions.png` | GT vs 預測 |
| `eval_<時間>.json` / `.log` | `evaluate.py` 的結果 |

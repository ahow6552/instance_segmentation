"""模型：單階段 U-Net（smp.Unet + 3 個 head）與 2 階段 stacked U-Net"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_head(in_ch, out_ch, mid_ch=64):
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=1),
    )


class InstanceHeads(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.semantic = conv_head(in_ch, num_classes)
        self.center = conv_head(in_ch, 1)
        self.offset = conv_head(in_ch, 2)

    def forward(self, x):
        return torch.cat([self.semantic(x), self.center(x), self.offset(x)], dim=1)


def build_unet(num_classes, encoder, encoder_weights, decoder_channels, in_channels=3):
    """smp.Unet，segmentation_head 換成 3 個 head；state_dict 跟 notebook 的 best_unet_instance.pth 相容"""
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        decoder_channels=decoder_channels,
    )
    model.segmentation_head = InstanceHeads(decoder_channels[-1], num_classes)
    return model


class StackedUNetInstance(nn.Module):
    """
    out1 = Stage1(image)
    out2 = out1 + Stage2([image, softmax(sem1), center1, offset1])
    Stage 2 最後一層初始化為 0 → 訓練一開始 out2 == out1
    """

    def __init__(self, num_classes, cfg):
        super().__init__()
        self.num_classes = num_classes
        out_ch = num_classes + 3
        self.stage1 = build_unet(num_classes, cfg["encoder"], cfg["encoder_weights"], cfg["decoder_channels"])
        self.stage2 = build_unet(num_classes, cfg["stage2_encoder"], cfg["stage2_encoder_weights"],
                                 cfg["decoder_channels"], in_channels=3 + out_ch)

        heads = self.stage2.segmentation_head
        for head in [heads.semantic, heads.center, heads.offset]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def stage2_input(self, x, out1):
        C = self.num_classes
        sem_prob = F.softmax(out1[:, :C].float(), dim=1).to(out1.dtype)   # logits 數值範圍大，轉成機率比較好學
        return torch.cat([x, sem_prob, out1[:, C:]], dim=1)

    def forward(self, x, return_all=False):
        out1 = self.stage1(x)
        out2 = out1 + self.stage2(self.stage2_input(x, out1))
        return (out1, out2) if return_all else out2


def build_model(cfg, num_classes, logger=None):
    log = logger.info if logger else print
    if cfg["model"] == "single":
        model = build_unet(num_classes, cfg["encoder"], cfg["encoder_weights"], cfg["decoder_channels"])
        log(f"model: single-stage U-Net ({cfg['encoder']})  params {count_params(model):.1f}M")
        return model

    model = StackedUNetInstance(num_classes, cfg)
    if cfg["stage1_init"]:
        model.stage1.load_state_dict(torch.load(cfg["stage1_init"], map_location="cpu"))
        log(f"stage 1 initialized from {cfg['stage1_init']}")
    log(f"model: stacked U-Net  stage1 {cfg['encoder']} {count_params(model.stage1):.1f}M + "
        f"stage2 {cfg['stage2_encoder']} {count_params(model.stage2):.1f}M = {count_params(model):.1f}M")
    return model


def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


def num_stages(model):
    return 2 if isinstance(model, StackedUNetInstance) else 1


def encoders(model):
    if isinstance(model, StackedUNetInstance):
        return [model.stage1.encoder, model.stage2.encoder]
    return [model.encoder]


def forward_all(model, images):
    """回傳每一階段的輸出 list：單階段 [out]，stacked [out1, out2]"""
    if isinstance(model, StackedUNetInstance):
        return list(model(images, return_all=True))
    return [model(images)]


def split_outputs(output, num_classes, offset_scale):
    return {
        "semantic": output[:, :num_classes],
        "center": output[:, num_classes:num_classes + 1],
        "offset": output[:, num_classes + 1:] * offset_scale,
    }

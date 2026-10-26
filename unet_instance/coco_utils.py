"""COCO 評估工具"""
import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def load_coco(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(path)


def coco_eval(coco_gt, results, img_ids, iou_type="segm", verbose=False):
    """回傳 COCOeval 物件；沒有任何偵測時回傳 None。verbose=True 會回傳 (E, summary 文字)"""
    if len(results) == 0:
        return (None, "no detections") if verbose else None
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
    E = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    E.params.imgIds = list(img_ids)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.evaluate(); E.accumulate(); E.summarize()
    return (E, buf.getvalue()) if verbose else E


def stats_dict(E):
    return dict(zip(STAT_NAMES, map(float, E.stats))) if E is not None else None


def per_class_ap(E, id_to_name):
    # precision: [IoU thresholds, recall, class, area range, max dets]
    precision = E.eval["precision"]
    out = {}
    for k, cat_id in enumerate(E.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        out[id_to_name[cat_id]] = float(p.mean()) if p.size else float("nan")
    return out

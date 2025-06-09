# utils/boxes.py
import math, torch, torchvision

# ---------- Koordinaten-Utilities ----------
def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2xywh(x):
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

# ---------- IoU-Berechnung (xywh-Format) ----------
def bbox_iou(box1, box2, eps=1e-7):
    """
    Intersection over Union für Tensoren in *Center-Format* [x,y,w,h].
    box1: (n,4), box2: (m,4)  →  gibt (n,m)-Matrix zurück
    """
    if box1.ndim == 1: box1 = box1[None, :]
    if box2.ndim == 1: box2 = box2[None, :]

    # Eckpunkte berechnen
    b1_xy1 = box1[:, :2] - box1[:, 2:] / 2
    b1_xy2 = box1[:, :2] + box1[:, 2:] / 2
    b2_xy1 = box2[:, :2] - box2[:, 2:] / 2
    b2_xy2 = box2[:, :2] + box2[:, 2:] / 2

    # Schnittmenge
    inter_xy1 = torch.max(b1_xy1[:, None, :], b2_xy1)          # (n,m,2)
    inter_xy2 = torch.min(b1_xy2[:, None, :], b2_xy2)          # (n,m,2)
    inter_wh  = (inter_xy2 - inter_xy1).clamp(min=0)           # (n,m,2)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]           # (n,m)

    # Flächen
    area1 = box1[:, 2] * box1[:, 3]                            # (n,)
    area2 = box2[:, 2] * box2[:, 3]                            # (m,)
    union = area1[:, None] + area2 - inter_area + eps

    return inter_area / union

# ---------- NMS ----------
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    assert prediction.ndim == 3
    dev = prediction.device
    bs  = prediction.shape[0]
    num_cls = prediction.shape[2] - 5
    output = [torch.zeros((0, 6), device=dev)] * bs
    for xi, x in enumerate(prediction):
        conf = x[:, 4:5] * x[:, 5:]
        x = torch.cat((x[:, :5], conf), 1)
        x = x[x[:, 5] > conf_thres]
        if not x.shape[0]:
            continue
        box = xywh2xyxy(x[:, 0:4])
        det = torch.cat((box, x[:, 5:6], torch.zeros_like(x[:, 5:6])), 1)
        keep = torchvision.ops.nms(det[:, :4], det[:, 4], iou_thres)
        output[xi] = det[keep[:max_det]]
    return output

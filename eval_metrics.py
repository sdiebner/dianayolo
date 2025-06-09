# eval_metrics.py  –  AP50-Evaluation (fix für “mAP = 0”)
# -------------------------------------------------------------
import torch, numpy as np
from utils.boxes import (non_max_suppression,
                         xywh2xyxy, xyxy2xywh, bbox_iou)

# -------------------------------------------------------------
# AP50 pro Klasse  (eine Klasse → Mittelwert = Wert selbst)
# -------------------------------------------------------------
def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-9):
    i = np.argsort(-conf)                                # sort by score (desc)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    uc = np.unique(target_cls)
    if len(uc) == 0:
        return 0.0, 0.0, 0.0

    p, r, ap = [], [], []
    for c in uc:
        j     = pred_cls == c
        n_gt  = (target_cls == c).sum()
        n_p   = j.sum()
        if n_p == 0 or n_gt == 0:
            p.append(0), r.append(0), ap.append(0); continue

        fpc = (1 - tp[j]).cumsum(0)
        tpc = tp[j].cumsum(0)

        recall    = tpc / (n_gt + eps)
        precision = tpc / (tpc + fpc + eps)

        ap.append((precision * recall).max())            # AP50
        p.append(precision[-1]), r.append(recall[-1])

    return float(np.mean(p)), float(np.mean(r)), float(np.mean(ap))

# -------------------------------------------------------------
# Haupt-Funktion  –  img_size  muss übergeben werden!
# -------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, img_size=640,
             conf_th=0.001, iou_th=0.5):
    """
    Berechnet P / R / AP50 auf dem übergebenen Dataloader.
    *img_size* = Quadrat-Side-Length, z. B. 640.
    """
    model.eval()
    stats = []                                           # (tp, conf, pred_cls, target_cls)

    strides = [4, 8, 16]                                 # wie im Training
    for imgs, t_list in dataloader:
        imgs = torch.stack(imgs).to(device)

        # ---- GT Tensor (n,6)  [img_idx, cls, x, y, w, h] (normiert 0-1) ----
        tgt = [torch.cat((torch.full((ti.shape[0],1), i), ti), 1)
               for i, ti in enumerate(t_list) if ti.numel()]
        targets = torch.cat(tgt, 0).to(device) if tgt else torch.zeros((0,6), device=device)

        # ---------------- Netzausgabe dekodieren -----------------
        preds_lvl = []
        for li, p in enumerate(model(imgs)):                         # 3 Ebenen
            bs,_,ny,nx = p.shape
            p = p.view(bs,3,model.num_outputs,ny,nx).permute(0,1,3,4,2).contiguous()

            gy,gx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
            grid   = torch.stack((gx,gy),2).view(1,1,ny,nx,2).to(device)
            anchor = model.anchors[li].to(device).view(1,3,1,1,2)

            xy  = (p[...,:2].sigmoid()*2 - 0.5 + grid) * strides[li]
            wh  = ((p[...,2:4].sigmoid()*2)**2) * anchor * strides[li]
            obj = p[...,4:5].sigmoid()
            cls = p[...,5:].sigmoid()

            preds_lvl.append(torch.cat((xy,wh,obj,cls), -1).view(bs,-1,model.num_outputs))

        detections = non_max_suppression(torch.cat(preds_lvl,1), conf_th, iou_th)

        # ---------------- Stats sammeln --------------------------
        for si, det in enumerate(detections):
            gt_img = targets[targets[:,0] == si, 1:]     # (n_gt,5)  [norm]
            correct = torch.zeros(det.shape[0], dtype=torch.bool, device=device)

            # ---- Ground-Truth in px-Koordinaten bringen ----
            if gt_img.numel():
                gt_boxes_px = xywh2xyxy(gt_img[:,1:5] * img_size)   # <-- FIX!
                for di, (*box_xyxy, conf, cls_id) in enumerate(det):
                    iou = bbox_iou(
                        xyxy2xywh(torch.tensor(box_xyxy, device=device)).unsqueeze(0),
                        xyxy2xywh(gt_boxes_px)
                    ).max(1)[0]
                    correct[di] = (iou > iou_th)

            stats.append((
                correct.cpu(),
                det[:,4].cpu(),
                det[:,5].cpu().int(),
                gt_img[:,0].cpu().int() if gt_img.numel() else torch.tensor([], dtype=torch.int)
            ))

    if not stats:
        return {'P':0, 'R':0, 'AP50':0, 'mAP50':0}

    tp, conf, pred_cls, target_cls = [torch.cat(x,0).numpy() for x in zip(*stats)]
    P, R, AP50 = ap_per_class(tp, conf, pred_cls, target_cls)
    return {'P':P, 'R':R, 'AP50':AP50, 'mAP50':AP50}

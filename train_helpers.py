# train_helpers.py  –  gemeinsame Utils für train.py
import torch, torch.nn as nn, math

# ---------------- Build Targets ----------------
def build_targets(preds, targets, anchors, strides):
    tcls, tbox, indices, anch = [], [], [], []
    nl, na = len(preds), anchors.shape[1]
    gain   = torch.ones(6, device=targets.device)
    if not targets.shape[0]:
        for _ in range(nl):
            indices.append((torch.tensor([]),)*4)
            tbox.append(torch.tensor([])); anch.append(torch.tensor([])); tcls.append(torch.tensor([]))
        return tcls, tbox, indices, anch
    for i in range(nl):
        gain[2:] = torch.tensor(preds[i].shape, device=targets.device)[[3,2,3,2]]
        t = targets * gain
        nt = t.shape[0]; t = t.repeat(na,1)
        ai = torch.arange(na, device=targets.device).repeat_interleave(nt)
        anchor_wh = anchors[i][ai]
        r = t[:,4:6]/anchor_wh
        j = (torch.max(r,1./r).max(1)[0] < 4.0)
        if not j.any():
            indices.append((torch.tensor([]),)*4)
            tbox.append(torch.tensor([])); anch.append(torch.tensor([])); tcls.append(torch.tensor([]))
            continue
        t, anchor_wh, ai = t[j], anchor_wh[j], ai[j]
        b, cls = t[:, :2].long().T
        gxy, gwh = t[:, 2:4], t[:, 4:6]
        gij = gxy.long(); gi, gj = gij.T
        indices.append((b, ai, gj, gi))
        tbox.append(torch.cat((gxy-gij, gwh), 1))
        anch.append(anchor_wh)
        tcls.append(cls)
    return tcls, tbox, indices, anch

# ---------------- IoU ----------------
def bbox_iou(box1, box2, eps=1e-7):
    (x1,y1,w1,h1), (x2,y2,w2,h2) = box1.T, box2.T
    inter = (torch.min(x1+w1/2, x2+w2/2) - torch.max(x1-w1/2, x2-w2/2)).clamp_(0) * \
            (torch.min(y1+h1/2, y2+h2/2) - torch.max(y1-h1/2, y2-h2/2)).clamp_(0)
    union = w1*h1 + w2*h2 - inter + eps
    return inter / union

# ---------------- Loss ----------------
def compute_loss(preds, targets, model, strides):
    device = preds[0].device
    BCE, MSE = nn.BCEWithLogitsLoss(), nn.MSELoss()
    anchors = model.anchors.to(device)
    tcls, tbox, indices, anch = build_targets(preds, targets, anchors, strides)
    lbox = lobj = lcls = torch.zeros(1, device=device)
    for i, p in enumerate(preds):
        bs,_,ny,nx = p.shape
        p = p.view(bs,3,model.num_outputs,ny,nx).permute(0,1,3,4,2).contiguous()
        b,a,gj,gi = indices[i]
        tobj = torch.zeros_like(p[...,0], device=device)
        if b.numel():
            pxy = (p[b,a,gj,gi,:2].sigmoid()*2 - 0.5)
            pwh = ((p[b,a,gj,gi,2:4].sigmoid()*2)**2) * anch[i]
            pbox = torch.cat((pxy,pwh),1)
            lbox += 0.05 * MSE(pbox, tbox[i])
            iou = bbox_iou(pbox, tbox[i]).detach()
            tobj[b,a,gj,gi] = iou.clamp_(0)
            if model.num_classes > 1:
                tc = torch.zeros_like(p[b,a,gj,gi,5:])
                tc[range(b.shape[0]), tcls[i]] = 1.
                lcls += 0.5 * BCE(p[b,a,gj,gi,5:], tc)
        lobj += 1.0 * BCE(p[...,4], tobj)
    return lbox + lobj + lcls

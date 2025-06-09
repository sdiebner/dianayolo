#!/usr/bin/env python3
# utils/kmeans_anchors.py
import argparse, numpy as np
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm

def load_boxes(mat_path):
    m = loadmat(mat_path)
    w, h = [], []
    for ev_idx, ev in enumerate(m["event_list"]):
        files = m["file_list"][ev_idx][0]
        bbxs  = m["face_bbx_list"][ev_idx][0]
        for f_idx in range(len(files)):
            boxes = bbxs[f_idx][0]       # (n,4)  x,y,w,h
            w.extend(boxes[:,2]); h.extend(boxes[:,3])
    return np.stack([w,h], 1)

def iou(box, clusters):
    x = np.minimum(box[0], clusters[:,0])
    y = np.minimum(box[1], clusters[:,1])
    inter = x*y
    return inter / (box[0]*box[1] + clusters[:,0]*clusters[:,1] - inter + 1e-9)

def kmeans(boxes, k=9, iters=100):
    clusters = boxes[np.random.choice(boxes.shape[0], k, replace=False)]
    for _ in range(iters):
        dist  = 1 - np.array([iou(b, clusters) for b in boxes])
        group = dist.argmin(1)
        new   = np.array([boxes[group==i].mean(0) for i in range(k)])
        if np.allclose(clusters, new, atol=1e-3): break
        clusters = new
    return clusters

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="WIDER_train/wider_face_split/wider_face_train.mat")
    ap.add_argument("--clusters", type=int, default=9)
    ap.add_argument("--img_size", type=int, default=640)
    args = ap.parse_args()

    boxes = load_boxes(args.mat)
    clust = kmeans(boxes, k=args.clusters)
    clust = clust[np.argsort(clust.prod(1))].round().astype(int)

    print("\nAnchors (raw pixels):")
    print(", ".join(f"[{w}, {h}]" for w,h in clust))

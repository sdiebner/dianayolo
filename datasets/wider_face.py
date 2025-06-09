
import os, numpy as np, torch, cv2
from scipy.io import loadmat
from pathlib import Path
from torch.utils.data import Dataset

class WIDERFaceDataset(Dataset):
    """
    Dataset-Loader für die .mat-Annotationen des WIDER-Face-Datensatzes.
    Gibt Bilder (FloatTensor CHW, [0,1]) und Targets (Tensor n x 5: cls,x,y,w,h) zurück.
    """

    def __init__(self, root_dir, split_mat, img_size=640, transform=None):
        self.root_dir = Path(root_dir)          # z.B. WIDER_train/images
        self.img_size = img_size
        self.transform = transform
        self.samples = []                       # List[Tuple[str, np.ndarray]]
        self._parse_mat(split_mat)

    def _parse_mat(self, split_mat):
        data = loadmat(split_mat)
        event_list = data['event_list']
        file_list = data['file_list']
        face_bbx_list = data['face_bbx_list']
        for ev_idx, ev in enumerate(event_list):
            event_name = ev[0][0]
            files = file_list[ev_idx][0]
            bbxs = face_bbx_list[ev_idx][0]
            for f_idx, f in enumerate(files):
                img_rel = os.path.join(event_name, f[0][0] + '.jpg')
                boxes = bbxs[f_idx][0]  # (n,4) x,y,w,h
                self.samples.append((img_rel, boxes.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def _resize_pad(self, img, boxes):
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            boxes = boxes * r
        h, w = img.shape[:2]
        top = (self.img_size - h) // 2
        left = (self.img_size - w) // 2
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[top:top+h, left:left+w] = img
        boxes[:, 0] += left
        boxes[:, 1] += top
        return canvas, boxes

    def __getitem__(self, idx):
        rel_path, boxes = self.samples[idx]
        img_path = self.root_dir / rel_path
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, boxes = self._resize_pad(img, boxes.copy())

        # xywh normalisiert
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / self.img_size
        boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / self.img_size
        boxes_xywh[:, 2] = boxes[:, 2] / self.img_size
        boxes_xywh[:, 3] = boxes[:, 3] / self.img_size

        labels = np.zeros((boxes_xywh.shape[0], 1), dtype=np.float32)  # <-- dtype hinzufügen
        targets = np.concatenate((labels, boxes_xywh), 1)

        if self.transform:
            img, targets = self.transform(img, targets)

        img = (img.astype(np.float32) / 255.).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(targets)

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

from datasets import ConjunctivaDataset
from unet import UNet


def map_bbox_to_original(bbox, meta):
    (w, h) = meta["orig_size"]
    scale = meta["scale"]
    pad_left, pad_top = meta["pad"]
    x1, y1, x2, y2 = bbox
    x1 = (x1 - pad_left) / scale
    x2 = (x2 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    y2 = (y2 - pad_top) / scale
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    return [int(x1), int(y1), int(x2), int(y2)]


def inference(img_root: str, ckpt: str, out_dir: str, img_size: int):
    dataset = ConjunctivaDataset(img_root, mask_root=None, mask_suffix=None, img_size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_c=3, out_c=1).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for imgs, meta in loader:
        imgs = imgs.to(device)
        meta = {k: v[0] for k, v in meta.items()}
        with torch.no_grad():
            preds = model(imgs).sigmoid()[0, 0]
        mask = (preds > 0.5).cpu().numpy().astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            bbox_model = [0, 0, 0, 0]
        else:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bbox_model = [x1, y1, x2, y2]
        bbox = map_bbox_to_original(bbox_model, meta)
        name = Path(meta["path"]).stem
        with open(out_path / f"{name}_bbox.json", "w") as f:
            json.dump({"bbox": bbox}, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and save bounding boxes")
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(args.img_root, args.ckpt, args.out_dir, args.img_size)

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def resize_with_padding(img: Image.Image, size: int, interpolation=Image.BILINEAR):
    """Resize image keeping aspect ratio and pad to square."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), interpolation)
    new_img = Image.new(img.mode, (size, size))
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    new_img.paste(img, (pad_left, pad_top))
    return new_img, scale, (pad_left, pad_top)


class ConjunctivaDataset(Dataset):
    """Dataset for conjunctival ROI segmentation or inference."""

    def __init__(
        self,
        img_root: str,
        mask_root: Optional[str] = None,
        mask_suffix: Optional[str] = None,
        img_size: int = 256,
    ) -> None:
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.has_mask = self.mask_root is not None and self.mask_suffix is not None

        self.samples: List[Tuple[Path, Optional[Path]]] = []
        for img_path in sorted(self.img_root.rglob("*.jpg")):
            if self.has_mask:
                if self.mask_root == self.img_root:
                    mask_path = img_path.with_name(img_path.stem + self.mask_suffix)
                else:
                    mask_path = self.mask_root / (img_path.stem + self.mask_suffix)
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))
            else:
                self.samples.append((img_path, None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size
        img_resized, scale, pad = resize_with_padding(img, self.img_size)
        img_tensor = TF.to_tensor(img_resized)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask_resized, _, _ = resize_with_padding(mask, self.img_size, Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).unsqueeze(0).float()
            return img_tensor, mask_tensor
        else:
            meta = {
                "orig_size": orig_size,
                "scale": scale,
                "pad": pad,
                "path": img_path,
            }
            return img_tensor, meta

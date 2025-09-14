import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from datasets import ConjunctivaDataset
from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for conjunctiva segmentation")
    parser.add_argument("--img-root", nargs="+", required=True, help="Image root directories")
    parser.add_argument("--mask-root", nargs="+", required=True, help="Mask root directories")
    parser.add_argument("--mask-suffix", nargs="+", required=True, help="Mask suffix for each dataset")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True, help="Output checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    assert len(args.img_root) == len(args.mask_root) == len(args.mask_suffix)
    datasets = []
    for img_root, mask_root, mask_suffix in zip(args.img_root, args.mask_root, args.mask_suffix):
        datasets.append(ConjunctivaDataset(img_root, mask_root, mask_suffix, args.img_size))
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_c=3, out_c=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()

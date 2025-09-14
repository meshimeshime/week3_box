# Conjunctival ROI Pipeline

This repository provides a minimal pipeline to train a U-Net on conjunctival ROI masks and
apply the trained model to generate bounding boxes for new eye images. Three datasets are
assumed:

1. **Conjunctival Images for Anemia Detection** – contains paired eye images and palpebral masks.
2. **Eye Conjunctiva Segmentation Dataset** – images and masks are stored in separate folders.
3. **Palpebral Conjunctiva Dataset** – only eye images, used for inference.

## Dataset layouts

```
Conjunctival Images for Anemia Detection/
├── India/1/20200124_155418.jpg
├── India/1/20200124_155418_palpebral.png
└── ...

Eye Conjunctiva Segmentation Dataset/
├── images/img_001.jpg
└── masks/img_001.png

Palpebral Conjunctiva Dataset/
├── img_1_001.jpg
└── ...
```

## Training

`train.py` accepts multiple datasets at once. For each dataset specify the image root,
mask root and mask suffix. Aspect ratio is preserved with padding during resizing.

Example (using the first dataset only):

```
python train.py --img-root "Conjunctival Images for Anemia Detection" \
                --mask-root "Conjunctival Images for Anemia Detection" \
                --mask-suffix "_palpebral.png" \
                --epochs 50 --out stage1.pt
```

Training with both mask datasets:

```
python train.py --img-root "Conjunctival Images for Anemia Detection" \
                             "Eye Conjunctiva Segmentation Dataset/images" \
                --mask-root "Conjunctival Images for Anemia Detection" \
                             "Eye Conjunctiva Segmentation Dataset/masks" \
                --mask-suffix "_palpebral.png" ".png" \
                --epochs 50 --out stage1.pt
```

## Inference & Bounding boxes

Apply a trained checkpoint to images that have no mask to obtain bounding boxes.
The bounding boxes are saved as `*_bbox.json` suitable as prompts for SlimSAM.

```
python inference_bbox.py --img-root "Palpebral Conjunctiva Dataset" \
                         --ckpt stage1.pt \
                         --out-dir preds_bbox \
                         --img-size 256
```

Each JSON file contains

```json
{"bbox": [x1, y1, x2, y2]}
```

where coordinates are in the original image size.

## Modules
- `datasets.py` – dataset loader with optional masks and aspect ratio preserving resize.
- `unet.py` – U-Net architecture (in_channels=3, out_channels=1).
- `train.py` – training script saving a `.pt` checkpoint.
- `inference_bbox.py` – generate bounding boxes from predicted masks.

## License
MIT

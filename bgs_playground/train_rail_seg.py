#!/usr/bin/env python3
"""Segmentation approach for rail detection.

Instead of predicting K slots × R rows, we predict a binary heatmap:
for each pixel, is it on a rail line or not?

This is MUCH simpler — no slot assignment, no ordering.
Post-processing extracts individual lines from the segmentation mask.

Architecture: ResNet-34 encoder + lightweight decoder (UNet-style skip connections)
"""

import argparse
import json
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models


IMG_POOL_DIRS = [
]


def _resolve_item_image(item_dir: Path | None, item_id: str, image_name: str | None = None):
    if item_dir is not None:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = item_dir / f"image{ext}"
            if p.exists():
                return p
    for pool in IMG_POOL_DIRS:
        if image_name:
            p = pool / image_name
            if p.exists():
                return p
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = pool / f"{item_id}{ext}"
            if p.exists():
                return p
    return None


def _parse_cvat_points(points: str) -> list[list[float]]:
    result = []
    for pair in points.split(";"):
        if not pair:
            continue
        x, y = pair.split(",", 1)
        result.append([float(x), float(y)])
    return result


def load_cvat_xml(cvat_xml: Path) -> tuple[dict[str, dict[str, list]], dict[str, str]]:
    """Load CVAT image annotation XML into in-memory polyline data."""
    root = ET.parse(cvat_xml).getroot()
    polyline_data: dict[str, dict[str, list]] = {}
    image_names: dict[str, str] = {}

    for image in root.findall("image"):
        name = image.attrib.get("name")
        if not name:
            continue
        item_id = Path(name).stem
        image_names[item_id] = Path(name).name
        grouped: dict[str, list] = {}
        for polyline in image.findall("polyline"):
            label = polyline.attrib.get("label", "rail")
            points = _parse_cvat_points(polyline.attrib.get("points", ""))
            if len(points) >= 2:
                grouped.setdefault(label, []).append(points)
        if grouped:
            polyline_data[item_id] = grouped

    return polyline_data, image_names


def load_native_polylines(labels_dir: Path, mask_set: str) -> dict[str, dict[str, list]]:
    """Load repo-native annotations/items/<id>/_sets/<mask_set>/polylines.json."""
    polyline_data: dict[str, dict[str, list]] = {}
    for d in sorted(labels_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        pfile = d / "_sets" / mask_set / "polylines.json"
        if not pfile.exists():
            continue
        data = json.loads(pfile.read_text())
        if any(len(lines) > 0 for lines in data.values()):
            polyline_data[d.name] = data
    return polyline_data


class RailSegDataset(Dataset):
    """Generate heatmap targets from polyline vertices."""

    def __init__(self, labels_dir: Path | None, mask_set: str, img_size: tuple,
                 item_ids: list, augment: bool = False, line_width: int = 6,
                 classes: list | None = None, augment_mode: str = "default",
                 polyline_data: dict[str, dict[str, list]] | None = None,
                 image_names: dict[str, str] | None = None):
        self.labels_dir = labels_dir
        self.mask_set = mask_set
        self.img_w, self.img_h = img_size
        self.items = item_ids
        self.augment = augment
        self.line_width = line_width
        self.classes = classes  # None = merge all into 1 channel; list = one channel per class
        self.augment_mode = augment_mode
        self.polyline_data = polyline_data
        self.image_names = image_names or {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_id = self.items[idx]
        item_dir = self.labels_dir / item_id if self.labels_dir else None

        # Load image (item dir first, then shared image pool)
        img = None
        img_path = _resolve_item_image(item_dir, item_id, self.image_names.get(item_id))
        if img_path is not None:
            img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"No image found for item {item_id}")
        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h))

        # Load polylines and render heatmap(s)
        data = None
        if self.polyline_data is not None:
            data = self.polyline_data.get(item_id)
        elif item_dir is not None:
            pfile = item_dir / "_sets" / self.mask_set / "polylines.json"
            if pfile.exists():
                data = json.loads(pfile.read_text())

        num_channels = len(self.classes) if self.classes else 1
        heatmap = np.zeros((num_channels, self.img_h, self.img_w), dtype=np.float32)
        if data:
            for cls_name, lines in data.items():
                if self.classes:
                    if cls_name not in self.classes:
                        continue
                    ch = self.classes.index(cls_name)
                else:
                    ch = 0
                for line in lines:
                    if len(line) < 2:
                        continue
                    pts = np.array(line, dtype=np.float32)
                    pts[:, 0] *= self.img_w / orig_w
                    pts[:, 1] *= self.img_h / orig_h
                    pts = pts.astype(np.int32)
                    for i in range(len(pts) - 1):
                        cv2.line(heatmap[ch], tuple(pts[i]), tuple(pts[i + 1]),
                                 1.0, self.line_width, cv2.LINE_AA)

        if self.augment:
            img, heatmap = self._augment(img, heatmap)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        return torch.from_numpy(img), torch.from_numpy(heatmap)  # heatmap is (C, H, W)

    def _augment(self, img, heatmap):
        """Apply spatial + photometric augmentations.

        heatmap is (C, H, W). Spatial transforms applied per-channel.
        """
        h, w = img.shape[:2]
        n_ch = heatmap.shape[0]

        # Horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            for c in range(n_ch):
                heatmap[c] = cv2.flip(heatmap[c], 1)

        # Horizontal shift
        if random.random() < 0.5:
            shift = int(random.uniform(-0.15, 0.15) * w)
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            for c in range(n_ch):
                heatmap[c] = cv2.warpAffine(heatmap[c], M, (w, h))

        # Vertical shift
        if random.random() < 0.4:
            shift = int(random.uniform(-0.1, 0.1) * h)
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            for c in range(n_ch):
                heatmap[c] = cv2.warpAffine(heatmap[c], M, (w, h))

        # Scale
        if random.random() < 0.4:
            scale = random.uniform(0.85, 1.15)
            new_w, new_h = int(w * scale), int(h * scale)
            s_img = cv2.resize(img, (new_w, new_h))
            if scale > 1:
                cy, cx = new_h // 2, new_w // 2
                img = s_img[cy - h//2:cy - h//2 + h, cx - w//2:cx - w//2 + w]
                for c in range(n_ch):
                    s_heat = cv2.resize(heatmap[c], (new_w, new_h))
                    heatmap[c] = s_heat[cy - h//2:cy - h//2 + h, cx - w//2:cx - w//2 + w]
            else:
                canvas_img = np.zeros_like(img)
                py, px = (h - new_h) // 2, (w - new_w) // 2
                canvas_img[py:py+new_h, px:px+new_w] = s_img
                img = canvas_img
                for c in range(n_ch):
                    s_heat = cv2.resize(heatmap[c], (new_w, new_h))
                    canvas_heat = np.zeros((h, w), dtype=np.float32)
                    canvas_heat[py:py+new_h, px:px+new_w] = s_heat
                    heatmap[c] = canvas_heat

        # Brightness/contrast
        if random.random() < 0.7:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-30, 30)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Hue/saturation
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Noise
        if random.random() < 0.3:
            noise = np.random.randn(*img.shape).astype(np.float32) * random.uniform(5, 15)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Blur
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        if self.augment_mode == "strong":
            # Small camera/vehicle attitude changes. Keep these mild so rails
            # remain plausible while exposing the model to labeling viewpoint noise.
            if random.random() < 0.45:
                angle = random.uniform(-3.0, 3.0)
                scale = random.uniform(0.97, 1.03)
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                for c in range(n_ch):
                    heatmap[c] = cv2.warpAffine(heatmap[c], M, (w, h))

            # Simulate partial exposure/occlusion variation without modifying labels.
            if random.random() < 0.35:
                x0 = random.randint(0, max(0, w - 1))
                y0 = random.randint(0, max(0, h - 1))
                cw = random.randint(max(8, w // 24), max(9, w // 8))
                ch = random.randint(max(8, h // 24), max(9, h // 8))
                x1 = min(w, x0 + cw)
                y1 = min(h, y0 + ch)
                fill = np.array([random.randint(20, 235) for _ in range(3)], dtype=np.uint8)
                alpha = random.uniform(0.35, 0.7)
                img[y0:y1, x0:x1] = (
                    img[y0:y1, x0:x1].astype(np.float32) * (1.0 - alpha)
                    + fill.astype(np.float32) * alpha
                ).astype(np.uint8)

            if random.random() < 0.45:
                gamma = random.uniform(0.75, 1.35)
                lut = np.array([((i / 255.0) ** gamma) * 255.0
                                for i in range(256)], dtype=np.uint8)
                img = cv2.LUT(img, lut)

        heatmap = np.clip(heatmap, 0, 1)
        return img, heatmap


class RailSegModel(nn.Module):
    """Simple encoder-decoder for rail segmentation."""

    def __init__(self, freeze_backbone: int = 6, num_classes: int = 1,
                 backbone_name: str = "resnet34"):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        if backbone_name == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            ch1, ch2, ch3, ch4 = 64, 128, 256, 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            ch1, ch2, ch3, ch4 = 256, 512, 1024, 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64ch
        self.pool = backbone.maxpool  # /4
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        if freeze_backbone > 0:
            layers = [self.enc0, self.pool, self.enc1, self.enc2, self.enc3, self.enc4]
            for i in range(min(freeze_backbone, len(layers))):
                for p in layers[i].parameters():
                    p.requires_grad = False

        # Decoder — channels parametrized by encoder output widths
        self.up4 = nn.Sequential(
            nn.Conv2d(ch4, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.Conv2d(256 + ch3, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + ch2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(
            nn.Conv2d(64 + ch1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up0 = nn.Sequential(
            nn.Conv2d(32 + 64, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)        # /2
        e1 = self.enc1(self.pool(e0))  # /4
        e2 = self.enc2(e1)       # /8
        e3 = self.enc3(e2)       # /16
        e4 = self.enc4(e3)       # /32

        # Decoder with skip connections
        d4 = self.up4(e4)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up3(torch.cat([d4, e3], dim=1))
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up2(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up1(torch.cat([d2, e1], dim=1))
        d1 = F.interpolate(d1, size=e0.shape[2:], mode='bilinear', align_corners=False)
        d0 = self.up0(torch.cat([d1, e0], dim=1))
        d0 = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.final(d0)  # (B, C, H, W) — logits, C = num_classes


def dice_loss(pred, target, smooth=1.0):
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum(dim=(2, 3))
    union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path,
                        help="Repo-native annotations/items directory.")
    parser.add_argument("--cvat-xml", type=Path,
                        help="CVAT XML annotation export. If set, labels-dir is optional.")
    parser.add_argument("--mask-set", default="rails")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-pool-dir", action="append", default=[],
                        help="Directory containing <item_id> image files. May be provided multiple times.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 360])
    parser.add_argument("--line-width", type=int, default=6)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--freeze-backbone", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="Weight for positive class in BCE (rails are sparse)")
    parser.add_argument("--classes", type=str, nargs="*", default=None,
                        help="Class names for multi-channel output. One channel per class.")
    parser.add_argument("--augment-mode", choices=["default", "strong"], default="default")
    args = parser.parse_args()
    IMG_POOL_DIRS[:] = [Path(p) for p in args.image_pool_dir]

    if args.cvat_xml is None and args.labels_dir is None:
        parser.error("one of --labels-dir or --cvat-xml is required")

    labels_dir = args.labels_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_w, img_h = args.img_size
    num_classes = len(args.classes) if args.classes else 1

    if args.cvat_xml is not None:
        polyline_data, image_names = load_cvat_xml(args.cvat_xml)
        annotation_format = "cvat_xml"
    else:
        polyline_data = load_native_polylines(labels_dir, args.mask_set)
        image_names = {}
        annotation_format = "native_polylines"

    items = []
    for item_id, data in sorted(polyline_data.items()):
        item_dir = labels_dir / item_id if labels_dir else None
        has_lines = any(len(lines) > 0 for lines in data.values())
        if has_lines and _resolve_item_image(item_dir, item_id, image_names.get(item_id)) is not None:
            items.append(item_id)

    random.seed(42)
    random.shuffle(items)
    n_val = max(1, int(len(items) * args.val_split))
    val_ids = items[:n_val]
    train_ids = items[n_val:]

    print(f"[SEG] Image: {img_w}x{img_h}, line_width={args.line_width}, classes={args.classes or ['all']}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RailSegDataset(labels_dir, args.mask_set, (img_w, img_h),
                               train_ids, augment=True, line_width=args.line_width,
                               classes=args.classes, augment_mode=args.augment_mode,
                               polyline_data=polyline_data, image_names=image_names)
    val_ds = RailSegDataset(labels_dir, args.mask_set, (img_w, img_h),
                             val_ids, augment=False, line_width=args.line_width,
                             classes=args.classes, polyline_data=polyline_data,
                             image_names=image_names)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = RailSegModel(freeze_backbone=args.freeze_backbone,
                         num_classes=num_classes).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} ({trainable:,} trainable)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    pos_weight = torch.tensor([args.pos_weight]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            pred = model(imgs)
            loss = bce(pred, targets) + dice_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = model(imgs)
                loss = bce(pred, targets) + dice_loss(pred, targets)
                val_loss += loss.item() * imgs.size(0)
                # Compute dice score
                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                inter = (pred_bin * targets).sum()
                union = pred_bin.sum() + targets.sum()
                val_dice += (2 * inter / (union + 1e-6)).item() * imgs.size(0)
        val_loss /= len(val_ds)
        val_dice /= len(val_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | dice {val_dice:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val:.4f}")

    # Save config
    config = {
        "model_type": "segmentation",
        "img_size": [img_w, img_h],
        "line_width": args.line_width,
        "freeze_backbone": args.freeze_backbone,
        "num_classes": num_classes,
        "classes": args.classes,
        "mask_set": args.mask_set,
        "annotation_format": annotation_format,
        "cvat_xml": str(args.cvat_xml) if args.cvat_xml else None,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "best_val_loss": best_val,
        "augment_mode": args.augment_mode,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()

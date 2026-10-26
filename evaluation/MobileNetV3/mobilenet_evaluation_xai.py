
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()
DATASET = ROOT / "dataset/BTXRD/dataset.xlsx"
MODEL_PATH = ROOT / "models/MobileNet-V3.pt"
IMAGE_DIR = ROOT / "dataset/BTXRD/images"
ANNOTATION_DIR = ROOT / "dataset/BTXRD/Annotations"
VAL_IDS = ROOT / "evaluation/val_ids.txt"
OUTPUT = ROOT / "evaluation/mobilenetv3_xai_metrics.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 640
IOU_PERCENTILE = 85
POINT_THRESHOLD = 0.5
AUC_STEPS = 20
BLUR_KERNEL = 31

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

AGE_MEAN = 35.57109479305741
AGE_STD = 20.997566907248853

print("Using device:", DEVICE)


# ============================================================
# DATA PREPROCESSING
# ============================================================

def padding(image, size=IMAGE_SIZE):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image, (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - new_height) // 2
    left = (size - new_width) // 2

    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def padding_mask(mask, size=IMAGE_SIZE):
    height, width = mask.shape[:2]
    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        mask, (new_width, new_height),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros((size, size), dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2

    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def preprocess_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=2, tileGridSize=(8, 8)
    ).apply(lab[:, :, 0])

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = padding(rgb).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD

    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def preprocess_context(row):
    age_z = (float(row["age"]) - AGE_MEAN) / AGE_STD

    context = np.array(
        [[
            age_z,
            float(row["gender_f"]),
            float(row["frontal"]),
            float(row["lateral"]),
            float(row["oblique"]),
        ]],
        dtype=np.float32,
    )

    return torch.from_numpy(context).to(DEVICE)


def load_metadata():
    df = pd.read_excel(DATASET)
    df["image_id"] = df["image_id"].astype(str)
    df["tumor"] = pd.to_numeric(df["tumor"], errors="coerce").fillna(0).astype(int)
    df["gender_f"] = (
        df["gender"].astype(str).str.strip().str[:1].str.upper().eq("F")
        .astype(np.float32)
    )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    for col in ["frontal", "lateral", "oblique"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    validation_ids = {
        line.strip()
        for line in VAL_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    df = df[df["image_id"].isin(validation_ids)].copy()
    df["image_path"] = df["image_id"].map(lambda x: IMAGE_DIR / x)
    df["annotation_path"] = df["image_id"].map(
        lambda x: (ANNOTATION_DIR / x).with_suffix(".json")
    )

    df = df[
        (df["tumor"] == 1)
        & df["age"].notna()
        & df["image_path"].map(Path.exists)
        & df["annotation_path"].map(Path.exists)
    ].reset_index(drop=True)

    return df


def resize_mask(json_path, mask_size):
    height, width = mask_size
    mask = np.zeros((height, width), dtype=np.uint8)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    polygons = [
        np.asarray(shape["points"], dtype=np.int32)
        for shape in data.get("shapes", [])
        if len(shape.get("points", [])) >= 3
    ]

    if polygons:
        cv2.fillPoly(mask, polygons, 1)

    return mask


# ============================================================
# MODEL: MATCH THE EXISTING FUSED CLASSIFIER
# ============================================================

class MobileNetV3Fused(nn.Module):
    def __init__(self, model_name="mobilenetv3_large_100", ctx_dim=5):
        super().__init__()

        self.encoder = timm.create_model(
            model_name, pretrained=False, features_only=True
        )

        channels = self.encoder.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(True),
        )

        self.fusion_head = nn.Linear(channels + 16, 1)

    def forward(self, image_tensor, context_tensor):
        features = self.encoder(image_tensor)[-1]
        pooled = self.pool(features).flatten(1)
        context_features = self.ctx_mlp(context_tensor)

        return self.fusion_head(
            torch.cat([pooled, context_features], dim=1)
        )


class ImageOnlyFusionWrapper(nn.Module):
    """
    Keeps context fixed while allowing Grad-CAM to differentiate
    with respect to the image input.
    """
    def __init__(self, model, context_tensor):
        super().__init__()
        self.model = model
        self.register_buffer("context_tensor", context_tensor.detach().clone())

    def forward(self, image_tensor):
        return self.model(image_tensor, self.context_tensor)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

    model = MobileNetV3Fused()

    checkpoint = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Unexpected checkpoint format.")

    state = checkpoint.get(
        "state_dict", checkpoint.get("model_state_dict", checkpoint)
    )

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")

    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }

    # The checkpoint has segmentation parameters in addition to the
    # fused-classification model. Load only the classification model's
    # matching keys, but require every model parameter to be present.
    model_state = model.state_dict()
    filtered = {
        key: value for key, value in state.items()
        if key in model_state
    }

    missing = sorted(set(model_state) - set(filtered))
    shape_mismatches = [
        key for key in filtered
        if filtered[key].shape != model_state[key].shape
    ]

    if missing or shape_mismatches:
        raise RuntimeError(
            "Checkpoint does not match the fused classifier.\n"
            f"Missing keys: {missing}\n"
            f"Shape mismatches: {shape_mismatches}"
        )

    model.load_state_dict(filtered, strict=True)
    model.to(DEVICE).eval()

    print("Loaded fused classifier weights.")
    print("Segmentation decoder weights are not used in this classification evaluation.")

    return model


# ============================================================
# LOCALIZATION METRICS
# ============================================================

def normalize_cam(cam):
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    cam -= cam.min()

    maximum = cam.max()
    if maximum > 1e-8:
        cam /= maximum

    return cam


def cam_metrics(cam_map, ground_truth):
    cam = normalize_cam(cam_map)
    gt = (ground_truth > 0).astype(np.uint8)

    threshold = np.percentile(cam, IOU_PERCENTILE)
    cam_binary = (cam >= threshold).astype(np.uint8)

    intersection = np.logical_and(gt, cam_binary).sum()
    union = np.logical_or(gt, cam_binary).sum()

    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (cam_binary.sum() + gt.sum() + 1e-8)

    hit = float(np.logical_and(gt, cam >= POINT_THRESHOLD).any())

    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    pointing = float(gt[peak_y, peak_x] > 0)

    return {
        "iou": float(iou),
        "hit_rate": hit,
        "dice": float(dice),
        "pointing_game": pointing,
    }


# ============================================================
# INSERTION / DELETION AUC
# ============================================================

def make_blurred_baseline(image_tensor):
    mean = torch.as_tensor(
        MEAN, dtype=image_tensor.dtype, device=image_tensor.device
    ).view(1, 3, 1, 1)

    std = torch.as_tensor(
        STD, dtype=image_tensor.dtype, device=image_tensor.device
    ).view(1, 3, 1, 1)

    rgb = (image_tensor * std + mean).clamp(0, 1)

    blurred = F.avg_pool2d(
        rgb, kernel_size=BLUR_KERNEL, stride=1,
        padding=BLUR_KERNEL // 2
    )

    return (blurred - mean) / std


def insertion_deletion_auc(model, image_tensor, context_tensor, cam_map):
    cam = normalize_cam(cam_map)

    cam_tensor = torch.from_numpy(cam).to(
        device=image_tensor.device, dtype=torch.float32
    )[None, None]

    cam_tensor = F.interpolate(
        cam_tensor,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    ranking = torch.argsort(cam_tensor.flatten(), descending=True)
    total_pixels = ranking.numel()

    baseline = make_blurred_baseline(image_tensor)
    original_flat = image_tensor.flatten(start_dim=2)
    baseline_flat = baseline.flatten(start_dim=2)

    insertion_scores = []
    deletion_scores = []

    with torch.no_grad():
        for step in range(AUC_STEPS + 1):
            count = int(round(total_pixels * step / AUC_STEPS))

            inserted = baseline_flat.clone()
            deleted = original_flat.clone()

            if count:
                indices = ranking[:count]
                inserted[:, :, indices] = original_flat[:, :, indices]
                deleted[:, :, indices] = baseline_flat[:, :, indices]

            inserted = inserted.reshape_as(image_tensor)
            deleted = deleted.reshape_as(image_tensor)

            insertion_scores.append(
                torch.sigmoid(model(inserted, context_tensor)).item()
            )
            deletion_scores.append(
                torch.sigmoid(model(deleted, context_tensor)).item()
            )

    x = np.linspace(0.0, 1.0, AUC_STEPS + 1)

    insertion_auc = float(np.trapezoid(insertion_scores, x))
    deletion_auc = float(np.trapezoid(deletion_scores, x))

    return insertion_auc, deletion_auc


# ============================================================
# EVALUATION
# ============================================================

def main():
    df = load_metadata()

    if df.empty:
        raise RuntimeError(
            "No validation tumor images with annotation JSONs were found."
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        previous = pd.read_csv(OUTPUT)
        completed = set(previous["image_id"].astype(str))
    else:
        previous = pd.DataFrame()
        completed = set()

    pending = df[~df["image_id"].isin(completed)]

    print("Eligible images:", len(df))
    print("Images remaining:", len(pending))

    if not pending.empty:
        model = load_model()

        for _, row in tqdm(
            pending.iterrows(), total=len(pending),
            desc="MobileNetV3 XAI"
        ):
            image_id = row["image_id"]
            image_bgr = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)

            if image_bgr is None:
                raise ValueError(f"Could not read {row['image_path']}")

            original_height, original_width = image_bgr.shape[:2]

            gt_original = resize_mask(
                row["annotation_path"], (original_height, original_width)
            )
            gt = padding_mask(gt_original)

            image_tensor = preprocess_image(image_bgr)
            context_tensor = preprocess_context(row)

            wrapper = ImageOnlyFusionWrapper(model, context_tensor).to(DEVICE).eval()

            target_layer = None
            for layer in wrapper.model.encoder.modules():
                if isinstance(layer, nn.Conv2d):
                    target_layer = layer

            if target_layer is None:
                raise RuntimeError("No convolutional target layer found.")

            methods = {
                "Grad_CAM": GradCAM,
                "Grad_CAMplusplus": GradCAMPlusPlus,
                "Layer_CAM": LayerCAM,
            }

            result = {"image_id": image_id}

            # CAM requires gradients; AUC forward passes are inference-only.
            with contextlib.ExitStack() as stack:
                cams = {
                    name: stack.enter_context(
                        cls(model=wrapper, target_layers=[target_layer])
                    )
                    for name, cls in methods.items()
                }

                for name, cam_method in cams.items():
                    cam_map = cam_method(
                        input_tensor=image_tensor,
                        targets=[ClassifierOutputTarget(0)],
                    )[0]

                    metrics = cam_metrics(cam_map, gt)
                    insertion_auc, deletion_auc = insertion_deletion_auc(
                        model, image_tensor, context_tensor, cam_map
                    )

                    for metric_name, value in metrics.items():
                        result[f"{name}_{metric_name}"] = value

                    result[f"{name}_insertion_auc"] = insertion_auc
                    result[f"{name}_deletion_auc"] = deletion_auc

            one_row = pd.DataFrame([result])

            if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
                one_row.to_csv(OUTPUT, mode="a", header=False, index=False)
            else:
                one_row.to_csv(OUTPUT, index=False)

    if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        results = pd.read_csv(OUTPUT)

        print("\n========== MOBILENETV3 XAI RESULTS ==========")
        print("Images saved:", len(results))

        for method in ["Grad_CAM", "Grad_CAMplusplus", "Layer_CAM"]:
            print(f"\n{method.replace('_', ' ')}")

            for metric in [
                "iou", "hit_rate", "dice", "pointing_game",
                "insertion_auc", "deletion_auc",
            ]:
                column = f"{method}_{metric}"

                if column not in results.columns:
                    continue

                value = results[column].mean()

                if metric in ["hit_rate", "pointing_game"]:
                    print(f"  {metric}: {value * 100:.2f}%")
                else:
                    print(f"  {metric}: {value:.4f}")

        print("\nCSV:", OUTPUT)


if __name__ == "__main__":
    main()

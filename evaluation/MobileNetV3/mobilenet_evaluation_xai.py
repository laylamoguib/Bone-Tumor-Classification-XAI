
import cv2
import json
import numpy
import pandas
import timm
import torch

from torch import nn
from pathlib import Path
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm.auto import tqdm

DATASET = Path.cwd() / "dataset/BTXRD/dataset.xlsx"
MODEL = Path.cwd() / "models/MobileNet-V3.pt"
OUTPUT = Path.cwd() / "evaluation/mobilenetv3_xai_metrics.csv"

DEVICE = torch.device("cpu")
torch.use_deterministic_algorithms(True)


# Preprocessing — unchanged
def preprocess_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=2, tileGridSize=(8, 8)
    ).apply(lab[:, :, 0])

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = padding(image).astype(numpy.float32) / 255

    image = (
        image - numpy.array([0.485, 0.456, 0.406], numpy.float32)
    ) / numpy.array([0.229, 0.224, 0.225], numpy.float32)

    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def padding(img, size=640):
    height, width = img.shape[:2]
    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        img, (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = numpy.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - new_height) // 2
    left = (size - new_width) // 2

    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def padding_mask(mask, size=640):
    height, width = mask.shape[:2]
    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        mask, (new_width, new_height),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = numpy.zeros((size, size), dtype=resized.dtype)
    top = (size - new_height) // 2
    left = (size - new_width) // 2

    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def resize_mask(json_path, mask_size):
    height, width = mask_size
    mask = numpy.zeros((height, width), numpy.uint8)

    data = json.loads(json_path.read_text(encoding="utf-8"))

    polygons = [
        numpy.asarray(shape["points"], numpy.int32)
        for shape in data.get("shapes", [])
        if len(shape.get("points", [])) >= 3
    ]

    if polygons:
        cv2.fillPoly(mask, polygons, 1)

    return mask


# Original IoU and hit rate + Dice and Pointing Game
def cam_metrics(cam_map, groundtruth_mask, iou=85, point_thr=0.5):
    cam = cam_map.astype(numpy.float32)

    # Original IoU calculation — unchanged
    threshold = numpy.percentile(cam, iou)
    cam_bin = (cam >= threshold).astype(numpy.uint8)

    inter = numpy.logical_and(groundtruth_mask, cam_bin).sum()
    union = numpy.logical_or(groundtruth_mask, cam_bin).sum()

    iou_val = inter / (union + 1e-8)

    # Original hit-rate calculation — unchanged
    cam = cam - cam.min()
    cam = cam / (cam.max() - cam.min())

    hit = float(
        numpy.logical_and(
            groundtruth_mask, cam >= point_thr
        ).any()
    )

    # Additional metric 1: Dice coefficient
    dice = (2 * inter) / (
        cam_bin.sum() + groundtruth_mask.sum() + 1e-8
    )

    # Additional metric 2: Pointing Game
    peak_y, peak_x = numpy.unravel_index(
        numpy.argmax(cam_map), cam_map.shape
    )
    pointing_game = float(groundtruth_mask[peak_y, peak_x] > 0)

    return float(iou_val), hit, float(dice), pointing_game


# Model — unchanged
class MobileNetV3ImageOnly(nn.Module):
    def __init__(self, model="mobilenetv3_large_100"):
        super().__init__()

        self.encoder = timm.create_model(
            model, pretrained=False, features_only=True
        )

        channel = self.encoder.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(channel, 1)

    def forward(self, image_tensor):
        feat = self.encoder(image_tensor)[-1]
        pooled = self.pool(feat).flatten(1)
        return self.cls_head(pooled)


# Evaluation
def main():
    df = pandas.read_excel(DATASET)

    df["image_id"] = df["image_id"].astype(str)
    df["image_path"] = [
        str(Path.cwd() / "dataset/BTXRD/images" / image_id)
        for image_id in df["image_id"]
    ]

    df["tumor"] = pandas.to_numeric(
        df.get("tumor", 0), errors="coerce"
    ).fillna(0).astype(int)

    validation_ids = {
        line.strip()
        for line in (
            Path.cwd() / "evaluation" / "val_ids.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    rows = []

    for _, row in df.iterrows():
        if row["image_id"] in validation_ids and int(row["tumor"]) == 1:
            annotation = (
                Path.cwd() / "dataset/BTXRD/Annotations" / row["image_id"]
            ).with_suffix(".json")

            if annotation.exists() and Path(row["image_path"]).exists():
                rows.append(row)

    evaluation_df = pandas.DataFrame(rows).reset_index(drop=True)

    model = MobileNetV3ImageOnly().to(DEVICE).eval()

    state = torch.load(MODEL, map_location=DEVICE)
    model.load_state_dict(state.get("state_dict", state), strict=False)

    # Original target-layer selection — unchanged
    target_layer = None

    for layer in model.encoder.modules():
        if isinstance(layer, nn.Conv2d):
            target_layer = layer

    if target_layer is None:
        raise RuntimeError("No convolutional target layer found.")

    cams = {
        "Grad-CAM": GradCAM(
            model=model, target_layers=[target_layer]
        ),
        "Grad-CAM++": GradCAMPlusPlus(
            model=model, target_layers=[target_layer]
        ),
        "Layer-CAM": LayerCAM(
            model=model, target_layers=[target_layer]
        ),
    }

    targets = [ClassifierOutputTarget(0)]
    metrics = []

    print("Evaluation in progress...")

    try:
        for _, row in tqdm(
            evaluation_df.iterrows(), total=len(evaluation_df)
        ):
            img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)

            if img is None:
                print(f"Skipping unreadable image: {row['image_id']}")
                continue

            height, width = img.shape[:2]

            annotation = (
                Path.cwd() / "dataset/BTXRD/Annotations" / row["image_id"]
            ).with_suffix(".json")

            ground_truth = resize_mask(
                annotation, (height, width)
            )

            image = preprocess_image(img)
            groundtruth_mask = padding_mask(
                ground_truth, int(image.shape[-1])
            )

            result = {"image_id": row["image_id"]}

            for xai, cam in cams.items():
                cam_map = cam(
                    input_tensor=image, targets=targets
                )[0]

                iou, hit, dice, pointing = cam_metrics(
                    cam_map, groundtruth_mask, 85, 0.5
                )

                result[f"{xai}_iou"] = iou
                result[f"{xai}_hit_rate"] = hit
                result[f"{xai}_dice"] = dice
                result[f"{xai}_pointing_game"] = pointing

            metrics.append(result)

            # Save after every image, so progress isn't lost.
            pandas.DataFrame(metrics).to_csv(OUTPUT, index=False)

    finally:
        for cam in cams.values():
            cam.activations_and_grads.release()

    if not metrics:
        print("No new images were evaluated.")
        return

    evaluated_images = pandas.DataFrame(metrics)

    print(f"\nResults saved to: {OUTPUT}")
    print(f"Images evaluated: {len(evaluated_images)}")

    for xai in ["Grad-CAM", "Grad-CAM++", "Layer-CAM"]:
        print(f"\n{xai}")

        for metric in [
            "iou", "hit_rate", "dice", "pointing_game"
        ]:
            mean = float(evaluated_images[f"{xai}_{metric}"].mean())

            if metric in ["hit_rate", "pointing_game"]:
                print(f"{metric}: {mean * 100:.2f}%")
            else:
                print(f"{metric}: {mean:.4f}")


if __name__ == "__main__":
    main()


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
MODEL = Path.cwd() / "models/EfficientNet-B3.pt"

# Use GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.use_deterministic_algorithms(True)


# -------------------- Preprocessing --------------------

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

    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


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


def resize_mask(json_path: Path, mask_size):
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


# -------------------- Metrics --------------------

def cam_metrics(cam_map, groundtruth_mask, percentile=85, point_thr=0.5):
    cam = cam_map.astype(numpy.float32)
    gt = (groundtruth_mask > 0)

    # Normalize heatmap safely
    cam = cam - cam.min()
    cam_range = cam.max()

    if cam_range > 0:
        cam = cam / cam_range
    else:
        cam = numpy.zeros_like(cam)

    # Create binary highlighted-region mask
    threshold = numpy.percentile(cam, percentile)
    cam_binary = cam >= threshold

    # IoU
    intersection = numpy.logical_and(gt, cam_binary).sum()
    union = numpy.logical_or(gt, cam_binary).sum()
    iou = intersection / (union + 1e-8)

    # Hit rate: any highlighted pixel overlaps the tumor
    hit = float(numpy.logical_and(gt, cam >= point_thr).any())

    # Dice score
    dice = (
        2.0 * intersection
        / (gt.sum() + cam_binary.sum() + 1e-8)
    )

    # Pointing-game accuracy: maximum activation lies inside the tumor
    max_point = numpy.unravel_index(numpy.argmax(cam), cam.shape)
    pointing_game = float(gt[max_point])

    return float(iou), float(hit), float(dice), float(pointing_game)


# -------------------- Model --------------------

class EfficientNetB3(nn.Module):
    def __init__(self, model="efficientnet_b3"):
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


# -------------------- Evaluation --------------------

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
    }

    rows = []

    for _, row in df.iterrows():
        if row["image_id"] not in validation_ids or int(row["tumor"]) != 1:
            continue

        annotation_path = (
            Path.cwd() / "dataset/BTXRD/Annotations" / row["image_id"]
        ).with_suffix(".json")

        if annotation_path.exists():
            rows.append(row)

    evaluation_df = pandas.DataFrame(rows).reset_index(drop=True)

    if evaluation_df.empty:
        print("No eligible validation images with tumor annotations found.")
        return

    model = EfficientNetB3().to(device).eval()

    state = torch.load(MODEL, map_location=device)
    model.load_state_dict(
        state.get("state_dict", state), strict=False
    )

    # Select the last convolutional layer
    target_layer = None
    for layer in model.encoder.modules():
        if isinstance(layer, nn.Conv2d):
            target_layer = layer

    if target_layer is None:
        raise RuntimeError("Could not find a convolutional target layer.")

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

    output_path = (
        Path.cwd() / "evaluation" / "efficientnet_xai_metrics.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = []

    print(f"Evaluating {len(evaluation_df)} images...")
    print("Metrics: IoU, Hit Rate, Dice, Pointing-Game")
    print("CAM methods: Grad-CAM, Grad-CAM++, Layer-CAM")

    try:
        for _, row in tqdm(
            evaluation_df.iterrows(),
            total=len(evaluation_df),
            desc="Evaluating",
            unit="img"
        ):
            img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)

            if img is None:
                print(f"Warning: Could not read {row['image_path']}")
                continue

            height, width = img.shape[:2]

            annotation_path = (
                Path.cwd() / "dataset/BTXRD/Annotations" / row["image_id"]
            ).with_suffix(".json")

            ground_truth = resize_mask(
                annotation_path, (height, width)
            )

            image = preprocess_image(img)
            groundtruth_mask = padding_mask(
                ground_truth, int(image.shape[-1])
            )

            metric = {"image_id": row["image_id"]}

            for xai, cam in cams.items():
                cam_map = cam(
                    input_tensor=image, targets=targets
                )[0]

                iou, hit, dice, pointing = cam_metrics(
                    cam_map, groundtruth_mask,
                    percentile=85, point_thr=0.5
                )

                metric[f"{xai}_iou"] = iou
                metric[f"{xai}_hit"] = hit
                metric[f"{xai}_dice"] = dice
                metric[f"{xai}_pointing_game"] = pointing

            metrics.append(metric)

            # Save after each completed image
            pandas.DataFrame(metrics).to_csv(
                output_path, index=False
            )

    finally:
        for cam in cams.values():
            cam.activations_and_grads.release()

    evaluated_images = pandas.DataFrame(metrics)

    if evaluated_images.empty:
        print("No images were successfully evaluated.")
        return

    print("\nMean results across evaluated images:")

    for xai in ["Grad-CAM", "Grad-CAM++", "Layer-CAM"]:
        print(f"\n{xai}")
        print(f"  IoU: {evaluated_images[f'{xai}_iou'].mean():.4f}")
        print(f"  Hit Rate: {evaluated_images[f'{xai}_hit'].mean():.1%}")
        print(f"  Dice Score: {evaluated_images[f'{xai}_dice'].mean():.4f}")
        print(
            f"  Pointing-Game Accuracy: "
            f"{evaluated_images[f'{xai}_pointing_game'].mean():.1%}"
        )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

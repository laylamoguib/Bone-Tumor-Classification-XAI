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

DATASET = Path.cwd()/"dataset/BTXRD/dataset.xlsx"
MODEL = Path.cwd()/"models/EfficientNet-B3.pt"

torch.use_deterministic_algorithms(True)

#preprocessing
def preprocess_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:,:,0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = padding(image).astype(numpy.float32)/255
    image = (image - numpy.array([0.485, 0.456, 0.406], numpy.float32)) / numpy.array([0.229, 0.224, 0.225], numpy.float32)
    tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(torch.device("cpu"))
    return tensor

def padding(img, size: int = 640):
    height, width = img.shape[:2]
    scale = min(size/width, size/height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = numpy.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas

def padding_mask(mask, size: int = 640):
    height, width = mask.shape[:2]
    scale = min(size/width, size/height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    canvas = numpy.zeros((size, size), dtype=resized.dtype)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas

def resize_mask(json_path: Path, mask_size):
    Height, Width = mask_size
    mask = numpy.zeros((Height, Width), numpy.uint8)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    polygons = [numpy.asarray(shape["points"], numpy.int32) for shape in data.get("shapes", []) if len(shape.get("points", [])) >= 3]
    cv2.fillPoly(mask, polygons, 1)
    return mask

def cam_metrics(cam_mask, groundtruth_mask, iou=85, point_thr=0.5):
    cam = cam_mask.astype(numpy.float32)
    #IoU
    mask = numpy.percentile(cam, iou)
    cam_bin = (cam >= mask).astype(numpy.uint8)
    inter = numpy.logical_and(groundtruth_mask, cam_bin).sum()
    union = numpy.logical_or(groundtruth_mask, cam_bin).sum()
    iou = inter / (union + 1e-8)
    #Hit
    cam = cam - cam.min()
    cam = cam / ((cam.max() - cam.min()))
    hit = 1.0 if numpy.logical_and(groundtruth_mask, cam >= point_thr).any() else 0.0
    return float(iou), float(hit)


#model
class EfficientNetB3(nn.Module):
    def __init__(self, model='efficientnet_b3'):
        super().__init__()
        self.encoder = timm.create_model(model, pretrained=False, features_only=True)
        channel = self.encoder.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(channel, 1)

    def forward(self, image_tensor):
        feat = self.encoder(image_tensor)[-1]
        pooled = self.pool(feat).flatten(1)
        return self.cls_head(pooled)

#evaluation
def main():
    df = pandas.read_excel(DATASET)
    df["image_id"] = df["image_id"].astype(str)
    df["image_path"] = [str(Path.cwd() / "dataset/BTXRD/images" / image_id) for image_id in df["image_id"]]
    df["tumor"] = pandas.to_numeric(df.get("tumor", 0), errors="coerce").fillna(0).astype(int)

    validation_ids = {ln.strip() for ln in (Path.cwd() / "evaluation" / "val_ids.txt").read_text(encoding="utf-8").splitlines()}
    rows = []
    for index, row in df.iterrows():
        if row["image_id"] in validation_ids and int(row["tumor"]) == 1:
            jp = (Path.cwd() / "dataset/BTXRD/Annotations" / row["image_id"]).with_suffix(".json")
            if jp.exists(): 
                rows.append(row)
    evaluation_df = pandas.DataFrame(rows).reset_index(drop=True)
    model = EfficientNetB3('efficientnet_b3').to(torch.device("cpu")).eval()
    state = torch.load(MODEL, map_location=torch.device("cpu"))
    model.load_state_dict(state.get("state_dict", state), strict=False)
    target_layer = None
    for m in model.encoder.modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m

    cams = {"Grad-CAM": GradCAM(model=model, target_layers=[target_layer]),"Grad-CAM++": GradCAMPlusPlus(model=model, target_layers=[target_layer]),"Layer-CAM": LayerCAM(model=model, target_layers=[target_layer]),}
    targets = [ClassifierOutputTarget(0)]

    print("Evaluation in progress...")
    metrics = []
    for index, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        ground_truth = resize_mask((Path.cwd() / "dataset/BTXRD/Annotations" / row["image_id"]).with_suffix(".json"), (height, width))
        image = preprocess_image(img)
        side = int(image.shape[-1])
        groundtruth_mask = padding_mask(ground_truth, side)

        metric = {"image_id": row["image_id"]}
        for xai, cam in cams.items():
            cam_map = cam(input_tensor=image, targets=targets)[0]
            iou, hit = cam_metrics(cam_map, groundtruth_mask, 85, 0.5)
            metric[f"{xai}_iou"] = iou
            metric[f"{xai}_hit"] = hit
        metrics.append(metric)

    evaluated_images = pandas.DataFrame(metrics)

    for xai in ["Grad-CAM", "Grad-CAM++", "Layer-CAM"]:
        iou = float(evaluated_images[f"{xai}_iou"].mean())
        hit_rate = float(evaluated_images[f"{xai}_hit"].mean()) * 100.0
        print(f"{xai}: IoU: {iou:.4f}, Hit Rate: {hit_rate:.1f}%")

if __name__ == "__main__":
    main()

import cv2
import numpy
import pandas
import timm
import torch
from torch import nn
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from ultralytics import YOLO

MODEL_NAME = 'EfficientNet-B3'
torch.use_deterministic_algorithms(True)

def padding(img: numpy.ndarray, size: int = 640) -> numpy.ndarray:
    height, width = img.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(img, (new_width, new_height))
    canvas = numpy.zeros((size, size, 3))
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas

def preprocess_image(image: numpy.ndarray) -> torch.Tensor:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = padding(image).astype(numpy.float32) / 255.0
    mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
    std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.device("cpu"))
    return tensor

def preprocess_context(row: pandas.Series) -> torch.Tensor:
    age_mean, age_std = 35.57109479305741, 20.997566907248853
    age_z = (float(row["age"]) - age_mean) / age_std
    ctx = numpy.array([ age_z, float(row["gender_f"]), float(row["frontal"]),float(row["lateral"]), float(row["oblique"])], dtype=numpy.float32)[None, :]  
    return torch.from_numpy(ctx).to(torch.device("cpu"))

def load_test_meta(xlsx_path: Path, images_directory: Path, test_txt: Path) -> pandas.DataFrame:
    df = pandas.read_excel(xlsx_path)
    df = df[df["image_id"].astype(str).isin( set(Path(test_txt).read_text(encoding="utf-8").split()))].copy()
    df["image_id"] = df["image_id"].astype(str)
    df["gender_f"] = df["gender"].astype(str).str.strip().str.upper().str.startswith("F").astype(numpy.float32)
    df["age"] = df["age"].astype(float)
    df["image_path"] = [str(images_directory / image_id) for image_id in df["image_id"]]
    return df

class TumorClassificationModel(nn.Module):
    def __init__(self, model_name='efficientnet_b3', ctx_dim=5):
        super().__init__()
        timm_model_name = model_name.lower().replace('-', '_')
        self.encoder = timm.create_model(timm_model_name, pretrained=False, features_only=True)
        ch = self.encoder.feature_info.channels()[-1]
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, 32), nn.ReLU(True),nn.BatchNorm1d(32), nn.Dropout(0.10),nn.Linear(32, 16), nn.ReLU(True))
        self.fusion_head = nn.Linear(ch + 16, 1)

    def forward(self, image_tensor: torch.Tensor, context_tensor: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(image_tensor)[-1]
        pooled = self.pool(feat).flatten(1)
        ctx = self.ctx_mlp(context_tensor)
        return self.fusion_head(torch.cat([pooled, ctx], dim=1))

def is_tumor_class(class_name: str) -> bool:
    name = class_name.lower().replace("_", "").replace(" ", "")
    return any(k in name for k in ["tumor", "benign", "malignant"])

@torch.inference_mode()
def get_yolo_prediction(yolo_model: YOLO, image_path: str) -> bool:
    preds = yolo_model.predict(image_path, imgsz=640, verbose=False, device=str(torch.device("cpu")))
    if not preds or not hasattr(preds[0], "boxes"):
        return False
    
    names = preds[0].names
    for box in preds[0].boxes:
        class_idx = int(box.cls.item())
        class_name = names.get(class_idx, "")
        if is_tumor_class(class_name):
            return True
    return False

def main():
    test_df = load_test_meta(Path.cwd() / "dataset/BTXRD/dataset.xlsx", Path.cwd() / "dataset/BTXRD/images", Path.cwd() / "evaluation/test_ids.txt")
    model = TumorClassificationModel(model_name=MODEL_NAME).to(torch.device("cpu")).eval()
    ckpt = torch.load(Path.cwd() / f"models/{MODEL_NAME}.pt", map_location=torch.device("cpu"))
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    yolo_model = YOLO(Path.cwd() / "models/YOLOv8.pt")    
    y_true, y_pred = [], []
    total_images = len(test_df)

    with torch.inference_mode():
        for index, row in tqdm(test_df.iterrows(), total=total_images, desc="Evaluating", unit="img"):
            ground_truth = int(row.tumor)
            image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            image_tensor = preprocess_image(image)
            context_tensor = preprocess_context(row)
            fused_output = model(image_tensor, context_tensor)
            classifier_pred = (torch.sigmoid(fused_output).item() >= 0.5)
            yolo_pred = get_yolo_prediction(yolo_model, row["image_path"])
            if classifier_pred == yolo_pred:
                y_true.append(ground_truth)
                y_pred.append(int(classifier_pred))

    agreed_count = len(y_true)
    agreed_percent = (agreed_count / total_images * 100.0) if total_images > 0 else 0
    accuracy_percent = (accuracy_score(y_true, y_pred) * 100.0) if agreed_count > 0 else 0
    
    print(f"Model: {MODEL_NAME:<20}, Agreed: {agreed_percent:7.1f}%, Accuracy: {accuracy_percent:9.1f}%")

if __name__ == "__main__":
    main()


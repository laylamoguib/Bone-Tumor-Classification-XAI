import cv2
import numpy
import pandas
import timm
import torch
from torch import nn
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)

DATASET = Path.cwd()/"dataset/BTXRD/dataset.xlsx"
MODEL = Path.cwd()/"models/ConvNeXt-Tiny.pt"
torch.use_deterministic_algorithms(True)

#preprocessing
def preprocess_image(image): 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = padding(image).astype(numpy.float32)/255
    image = (image - numpy.array([0.485, 0.456, 0.406], numpy.float32)) / numpy.array([0.229, 0.224, 0.225], numpy.float32)
    tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(torch.device("cpu"))
    return tensor

#metadata
def load_test_meta(xlsx_path: Path, images_directory: Path, test_txt: Path):
    df = pandas.read_excel(xlsx_path)
    df = df[df["image_id"].astype(str).isin(set(Path(test_txt).read_text(encoding="utf-8").split()))].copy()
    df["image_id"] = df["image_id"].astype(str)
    df["gender_f"] = df["gender"].astype(str).str.strip().str[:1].str.upper().eq("F").astype(numpy.float32)
    df["age"] = df["age"].astype(float)
    df["image_path"] = [str(Path(images_directory)/ image) for image in df["image_id"]]
    return df

def padding(img, size: int = 640):
    height, width = img.shape[:2]
    scale = min(size/width, size/height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(img, (new_width, new_height))
    canvas = numpy.zeros((size, size, 3))
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas

def preprocess_context(row: pandas.Series):
    age_z = (float(row["age"]) - 35.57109479305741) / (20.997566907248853)
    ctx = numpy.array([age_z, float(row["gender_f"]), float(row["frontal"]), float(row["lateral"]), float(row["oblique"])], dtype=numpy.float32)[None, :]
    return torch.from_numpy(ctx).to(torch.device("cpu"))

#model
class ConvNeXtFused(nn.Module):
    def __init__(self, model_name='convnext_tiny', ctx_dim=5):

        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=False, features_only=True)
        ch = self.encoder.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, 32), nn.ReLU(True),nn.BatchNorm1d(32), nn.Dropout(0.10),nn.Linear(32, 16), nn.ReLU(True))
        self.fusion_head = nn.Linear(ch + 16, 1)

    def forward(self, image_tensor, context_tensor):
        feat = self.encoder(image_tensor)[-1]
        pooled = self.pool(feat).flatten(1)
        ctx = self.ctx_mlp(context_tensor)
        return self.fusion_head(torch.cat([pooled, ctx], dim=1))

#evaluation
def main():

    test_df = load_test_meta(DATASET, Path.cwd()/"dataset/BTXRD/images", Path.cwd()/"evaluation/test_ids.txt")
    model = ConvNeXtFused().to(torch.device("cpu")).eval()
    state = torch.load(MODEL, map_location=torch.device("cpu"))
    model.load_state_dict(state.get("state_dict", state), strict=False)
    normals_all = test_df[test_df["tumor"] == 0]
    normals_pool = normals_all.reset_index(drop=True)

    print("Evaluation in progress...")
    for index, site in enumerate(['hand','ulna','radius','humerus','foot','tibia','fibula','femur','hip bone']):
        tumors_site = test_df[(test_df[site] == 1) & (test_df["tumor"] == 1)]
        tumors_sel  = tumors_site.sample(min(len(tumors_site), len(normals_pool)), random_state=42)
        normals_sel = normals_pool.sample(min(len(tumors_site), len(normals_pool)), random_state=42)
        eval_df = pandas.concat([tumors_sel, normals_sel]).reset_index(drop=True)

        labels, certainty = [], []
        with torch.inference_mode():
            for index, row in eval_df.iterrows():
                image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
                fused = model(preprocess_image(image), preprocess_context(row))
                certainty.append(torch.sigmoid(fused).item())
                labels.append(int(row.tumor))

        ground_truth_label = numpy.array(labels)
        classification_certainty = numpy.array(certainty)
        classification_label = (classification_certainty >= 0.3).astype(int)

        print(f"[{site.upper()}]")
        print(f"Precision (No Tumour): {precision_score(ground_truth_label, classification_label, pos_label=0):.2f}")
        print(f"Recall (No Tumour): {recall_score(ground_truth_label, classification_label, pos_label=0):.2f}")
        print(f"F1 (No Tumour): {f1_score(ground_truth_label, classification_label, pos_label=0):.2f}")
        print(f"Precision (Tumour): {precision_score(ground_truth_label, classification_label, pos_label=1):.2f}")
        print(f"Recall (Tumour): {recall_score(ground_truth_label, classification_label, pos_label=1):.2f}")
        print(f"F1 (Tumour): {f1_score(ground_truth_label, classification_label, pos_label=1):.2f}")
        print(f"Accuracy: {accuracy_score(ground_truth_label, classification_label):.2f}")
        print(f"ROC-AUC: {roc_auc_score(ground_truth_label, classification_certainty):.3f}\n")

if __name__ == "__main__":
    main()
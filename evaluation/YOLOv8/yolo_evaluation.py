import numpy
import pandas
import torch
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from tqdm.auto import tqdm

DATASET = Path.cwd() / "dataset/BTXRD/dataset.xlsx"
MODEL = Path.cwd() / "models/YOLOv8.pt"
torch.use_deterministic_algorithms(True)

#preprocessing
def load_test_meta(xlsx_path: Path, images_directory: Path, test_txt: Path):
    df = pandas.read_excel(xlsx_path)
    test_ids = set(Path(test_txt).read_text(encoding="utf-8").split())
    df = df[df["image_id"].astype(str).isin(test_ids)].copy()

    df["image_id"] = df["image_id"].astype(str)
    df["benign"] = pandas.to_numeric(df.get("benign", 0), errors="coerce").fillna(0).astype(int)
    df["malignant"] = pandas.to_numeric(df.get("malignant", 0), errors="coerce").fillna(0).astype(int)

    gt = []
    for b, m in zip(df["benign"].tolist(), df["malignant"].tolist()):
        if int(m) == 1: gt.append(2)
        elif int(b) == 1: gt.append(1)
        else: gt.append(0)
    df["gt3"] = gt

    df["image_path"] = [str(Path(images_directory) / image) for image in df["image_id"]]
    return df

def _names_dict(model):
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}

def _class_ids_for(names_dict, wanted_set):
    wanted = set(s.lower() for s in wanted_set)
    ids = {i for i, n in names_dict.items() if n.strip().lower() in wanted}
    if not ids:
        ids = {i for i, n in names_dict.items() if any(w in n.strip().lower() for w in wanted)}
    return ids

#evaluation
def main():
    test_df = load_test_meta(DATASET, Path.cwd() / "dataset/BTXRD/images", Path.cwd() / "evaluation/test_ids.txt")
    model = YOLO(str(MODEL))
    names = _names_dict(model)
    benign_ids = _class_ids_for(names, {"benign"})
    malignant_ids = _class_ids_for(names, {"malignant"})

    print("Evaluation in progress...")
    labels, certainty = [], []
    with torch.inference_mode():
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating", unit="img"):
            res = model.predict(row["image_path"], imgsz=640, conf=0.001, verbose=False, device="cpu")
            boxes = res[0].boxes

            p_b, p_m = 0.0, 0.0
            if boxes is not None and len(boxes) > 0:
                cls = boxes.cls.detach().cpu().numpy().astype(int)
                conf = boxes.conf.detach().cpu().numpy()
                if benign_ids:
                    mb = numpy.isin(cls, list(benign_ids))
                    if mb.any(): p_b = float(conf[mb].max())
                if malignant_ids:
                    mm = numpy.isin(cls, list(malignant_ids))
                    if mm.any(): p_m = float(conf[mm].max())
            
            certainty.append([1.0 - max(p_b, p_m), p_b, p_m])
            labels.append(int(row["gt3"]))

    ground_truth_label = numpy.array(labels)
    classification_certainty = numpy.array(certainty)
    max_tumour_certainty = classification_certainty[:, 1:3].max(axis=1)
    classification_label = numpy.where(max_tumour_certainty < 0.200, 0, 1 + classification_certainty[:, 1:3].argmax(axis=1))
    prec = precision_score(ground_truth_label, classification_label, labels=[0, 1, 2], average=None, zero_division=0)
    rec = recall_score(ground_truth_label, classification_label, labels=[0, 1, 2], average=None, zero_division=0)
    f1 = f1_score(ground_truth_label, classification_label, labels=[0, 1, 2], average=None, zero_division=0)

    print(f"Precision (No Tumour): {prec[0]:.2f}")
    print(f"Recall (No Tumour): {rec[0]:.2f}")
    print(f"F1 (No Tumour): {f1[0]:.2f}")
    print(f"Precision (Benign): {prec[1]:.2f}")
    print(f"Recall (Benign): {rec[1]:.2f}")
    print(f"F1 (Benign): {f1[1]:.2f}")
    print(f"Precision (Malignant): {prec[2]:.2f}")
    print(f"Recall (Malignant): {rec[2]:.2f}")
    print(f"F1 (Malignant): {f1[2]:.2f}")
    print(f"Accuracy: {accuracy_score(ground_truth_label, classification_label):.2f}")

if __name__ == "__main__":
    main()


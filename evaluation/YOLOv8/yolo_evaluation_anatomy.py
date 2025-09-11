import numpy
import pandas
import torch
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)

DATASET = Path.cwd()/"dataset/BTXRD/dataset.xlsx"
MODEL = Path.cwd()/"models/YOLOv8.pt"
torch.use_deterministic_algorithms(True)

#metadata
def load_test_meta(xlsx_path: Path, images_directory: Path, test_txt: Path):
    df = pandas.read_excel(xlsx_path)
    df = df[df["image_id"].astype(str).isin(set(Path(test_txt).read_text(encoding="utf-8").split()))].copy()
    df["image_id"] = df["image_id"].astype(str)
    df["benign"] = pandas.to_numeric(df.get("benign", 0), errors="coerce").fillna(0).astype(int)
    df["malignant"] = pandas.to_numeric(df.get("malignant", 0), errors="coerce").fillna(0).astype(int)
    df["tumor"] = ((df["benign"] == 1) | (df["malignant"] == 1)).astype(int) # 0/1
    df["image_path"] = [str(Path(images_directory) / image) for image in df["image_id"]]
    return df

def _names_dict(model):
    if isinstance(model.names, dict):
        return {int(k): str(v) for k, v in model.names.items()}
    return {i: str(n) for i, n in enumerate(model.names)}

def _class_ids_for(names_dict, wanted_set):
    ids = {i for i, n in names_dict.items() if n.strip().lower() in set(s.lower() for s in wanted_set)}
    if len(ids) == 0:
        ids = {i for i, n in names_dict.items() if any(w in n.strip().lower() for w in set(s.lower() for s in wanted_set))}
    return ids

#evaluation
def main():
    test_df = load_test_meta(DATASET, Path.cwd() / "dataset/BTXRD/images", Path.cwd() / "evaluation/test_ids.txt")
    names = _names_dict(YOLO(str(MODEL)))
    tumour_ids = _class_ids_for(names, {"tumor", "tumour", "benign", "malignant"})
    normals_pool = test_df[test_df["tumor"] == 0].reset_index(drop=True)

    print("Evaluation in progress...")
    for site in ['hand','ulna','radius','humerus','foot','tibia','fibula','femur','hip bone']:
        tumors_site = test_df[(test_df.get(site, 0) == 1) & (test_df["tumor"] == 1)]
        m = min(len(tumors_site), len(normals_pool))
        tumors_sel = tumors_site.sample(n=m, random_state=42)
        normals_sel = normals_pool.sample(n=m, random_state=42)
        eval_df = pandas.concat([tumors_sel, normals_sel]).reset_index(drop=True)

        labels, certainty = [], []
        with torch.inference_mode():
            for index, row in eval_df.iterrows():
                res = YOLO(str(MODEL)).predict(row["image_path"], imgsz=640, conf=0.001, verbose=False, device="cpu")
                boxes = res[0].boxes
                p_t = 0.0
                if boxes is not None and len(boxes) > 0 and len(tumour_ids) > 0:
                    cls = boxes.cls.detach().cpu().numpy().astype(int)
                    conf = boxes.conf.detach().cpu().numpy()
                    mask = numpy.isin(cls, list(tumour_ids))
                    if mask.any():
                        p_t = float(conf[mask].max())
                certainty.append(p_t)
                labels.append(int(row["tumor"]))

        ground_truth_label = numpy.array(labels)
        classification_certainty = numpy.array(certainty)
        classification_label = (classification_certainty >= 0.200).astype(int)

        print(f"[{site.upper()}]")
        print(f"Precision (No Tumour): {precision_score(ground_truth_label, classification_label, pos_label=0, zero_division=0):.2f}")
        print(f"Recall (No Tumour): {recall_score( ground_truth_label, classification_label, pos_label=0, zero_division=0):.2f}")
        print(f"F1 (No Tumour): {f1_score( ground_truth_label, classification_label, pos_label=0, zero_division=0):.2f}")
        print(f"Precision (Tumour): {precision_score(ground_truth_label, classification_label, pos_label=1, zero_division=0):.2f}")
        print(f"Recall (Tumour): {recall_score( ground_truth_label, classification_label, pos_label=1, zero_division=0):.2f}")
        print(f"F1 (Tumour): {f1_score( ground_truth_label, classification_label, pos_label=1, zero_division=0):.2f}")
        print(f"Accuracy: {accuracy_score(ground_truth_label, classification_label):.2f}")
        try:
            print(f"ROC-AUC: {roc_auc_score(ground_truth_label, classification_certainty):.3f}\n")
        except Exception:
            print("ROC-AUC: n/a\n")

if __name__ == "__main__":
    main()

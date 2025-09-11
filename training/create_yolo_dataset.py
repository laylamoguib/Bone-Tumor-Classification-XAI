import json
from pathlib import Path
import cv2
import pandas
from tqdm import tqdm
import albumentations

DATASET = Path.cwd() / "Dataset/BTXRD"

def read_image_ids(path: Path):
    with open(path) as file:
        return [line.strip() for line in file]

def process_and_save_image(source_path: Path, destination_path: Path):
    image = cv2.imread(str(source_path))
    lightness, color_a, color_b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    processed_image = cv2.cvtColor(
        cv2.merge((cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness), color_a, color_b)),
        cv2.COLOR_LAB2BGR
    )
    cv2.imwrite(str(destination_path), processed_image)
    return processed_image

def polygons_to_yolo(image_shape: tuple, polygons: list, label_path: Path):
    image_height, image_width = image_shape[:2]
    yolo_lines = []
    for points, class_id in polygons:
        normalized_points = [f"{point[0]/image_width:.6f} {point[1]/image_height:.6f}" for point in points]
        if len(normalized_points) >= 3:
            yolo_lines.append(f"{class_id} {' '.join(normalized_points)}")
    label_path.write_text("\n".join(yolo_lines))

if __name__ == "__main__":
    metadata_df = pandas.read_excel(DATASET / "dataset.xlsx", engine="openpyxl")
    metadata_df["tumor"] = (metadata_df["benign"] | metadata_df["malignant"]).astype(int)

    train_image_ids = read_image_ids(Path.cwd() / "evaluation/train_ids.txt")
    validation_image_ids = read_image_ids(Path.cwd() / "evaluation/val_ids.txt")

    train_metadata = metadata_df[metadata_df["image_id"].isin(train_image_ids)]
    validation_metadata = metadata_df[metadata_df["image_id"].isin(validation_image_ids)]

    for path in [Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/images/train",Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/images/val",Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/labels/train",Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/labels/val",]:
        path.mkdir(parents=True, exist_ok=True)

    transform_pipeline = albumentations.Compose([albumentations.HorizontalFlip(p=0.5),albumentations.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),albumentations.RandomScale(scale_limit=0.15, p=0.5),albumentations.RandomBrightnessContrast(p=0.5),],keypoint_params=albumentations.KeypointParams(format="xy", remove_invisible=False))

    splits = [("train",train_metadata,Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/images/train",Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/labels/train"),("validation",validation_metadata,Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/images/val",Path.cwd() / "Dataset/BTXRD_YOLO_MultiClass/labels/val")]

    for split_name, split_metadata, image_directory, label_directory in splits:
        for index, row in tqdm(split_metadata.iterrows(), total=len(split_metadata)):
            image_id = row["image_id"]
            image_path = DATASET / "images" / image_id
            image_stem = Path(image_id).stem

            processed_image = process_and_save_image(image_path, image_directory / image_id)
            label_path = label_directory / f"{image_stem}.txt"

            if row["tumor"] == 0:
                label_path.touch()
                continue

            annotation_path = DATASET / "Annotations" / f"{image_stem}.json"
            polygons = []
            if annotation_path.exists():
                annotation_data = json.loads(annotation_path.read_text())
                class_id = int(row["malignant"]) 
                for shape in annotation_data.get("shapes", []):
                    points = shape.get("points", [])
                    if len(points) >= 3:
                        polygons.append((points, class_id))

            polygons_to_yolo(processed_image.shape, polygons, label_path)

            if split_name == "train" and polygons:
                keypoints = [kp for polygon, _ in polygons for kp in polygon]
                for i in range(2):
                    augmented = transform_pipeline(image=processed_image, keypoints=keypoints)
                    augmented_image, augmented_keypoints = augmented["image"], augmented["keypoints"]

                    augmented_stem = f"{image_stem}_aug{i}"
                    cv2.imwrite(str(image_directory / f"{augmented_stem}.jpg"), augmented_image)

                    augmented_polygons, start_index = [], 0
                    for polygon_points, class_id in polygons:
                        num_points = len(polygon_points)
                        new_points = augmented_keypoints[start_index : start_index + num_points]
                        augmented_polygons.append((new_points, class_id))
                        start_index += num_points

                    polygons_to_yolo(augmented_image.shape, augmented_polygons, label_directory / f"{augmented_stem}.txt")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json


# Load the excel file
dataframe = pd.read_excel(os.path.join(os.path.join("processed_dataset", "BTXRD"), "dataset.xlsx"), engine='openpyxl')

def load_and_prepare_sample(row):
    image_path = os.path.join(os.path.join(os.path.join("processed_dataset", "BTXRD"), "images"), row['image_id'])
    json_path = os.path.join(os.path.join(os.path.join("processed_dataset", "BTXRD"), "Annotations"), row['image_id'].replace('.jpeg', '.json'))
    
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    mask = np.zeros((224, 224), dtype=np.float32)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
                data = json.load(f)
        for shape in data['shapes']:
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], color=1.0)
            
    image_normalised = (image / 255.0).astype(np.float32)
    return image_normalised, mask

if __name__ == '__main__':

    # Find sample images for each class
    tumor_samples = dataframe[dataframe['tumor'] == 1].sample(6, random_state=42)
    normal_samples = dataframe[dataframe['tumor'] == 0].sample(6, random_state=42)

    # Combine them into one list for plotting
    all_samples = pd.concat([tumor_samples, normal_samples])

    # Create the plot
    plt.figure(figsize=(16, 8))

    # Show examples from dataset
    for i, (index, row) in enumerate(all_samples.iterrows()):
        image, mask = load_and_prepare_sample(row)
        
        if image is None:
            continue
            
        # Create the red overlay
        overlay = np.stack([
            mask * 0.8,  
            np.zeros_like(mask), 
            np.zeros_like(mask)  
        ], axis=-1)

        plt.subplot(2, 6, i + 1)
        plt.imshow(image)
        plt.imshow(overlay, alpha=0.5)
        plt.title(f"Label: {'Tumor' if row['tumor'] == 1 else 'Normal'}")
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
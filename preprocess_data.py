import os
import pandas as pd
import numpy as np
import cv2
import json

def resize_and_pad(image, updated_size=(224, 224)):
    height, width = image.shape[:2] #Takes the image height and width, ignores color channel
    updated_height, updated_width = updated_size

    scale = min(updated_width / width, updated_height / height) #Determine the scaling factor
    new_width, new_height = int(width * scale), int(height * scale) #Image resizing to scale

    # Interpolation: Fixing pixel grid after resizing
    if scale < 1: #Shrinking
        interp = cv2.INTER_AREA

    else: # scale >= 1 #Enlarging 
        interp = cv2.INTER_LINEAR

    #change the size of an image    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interp)


    if len(image.shape) == 2: # Annotation Masks

        # Create black canvas
        padded_image = np.zeros(updated_size, dtype=resized_image.dtype)

    else: # Radiograph image

        # Create a black canvas
        padded_image = np.zeros((updated_height, updated_width, 3), dtype=resized_image.dtype)

    # Calculate padding amounts
    top_padding = (updated_height - new_height) // 2
    left_padding = (updated_width - new_width) // 2
    
    #Set image on canvas filling the remainder with zeros
    padded_image[top_padding : (top_padding + new_height), left_padding : (left_padding + new_width)] = resized_image

    return padded_image

# Create the new directories
os.makedirs(os.path.join("processed_dataset", "BTXRD", "images"), exist_ok=True)
os.makedirs(os.path.join("processed_dataset", "BTXRD", "Annotations"), exist_ok=True)

# Load the metadata
dataframe = pd.read_excel(os.path.join(os.path.join("dataset", "BTXRD"), "dataset.xlsx"), engine='openpyxl')

for index, row in dataframe.iterrows():

    # Define paths for the input files
    raw_image_path = os.path.join(os.path.join(os.path.join("dataset", "BTXRD"), "images"), row['image_id'])
    base_filename_with_ext = row['image_id']
    json_filename = base_filename_with_ext.replace('.jpeg', '.json')
    raw_json_path = os.path.join(os.path.join(os.path.join("dataset", "BTXRD"), "Annotations"), json_filename)

    # Load the raw image
    image = cv2.imread(raw_image_path)
    original_h, original_w, index = image.shape

    # Apply the resizing and padding
    image_processed = resize_and_pad(image, updated_size=(224, 224))

    # Define the output path
    out_image_path = os.path.join(os.path.join("processed_dataset", "BTXRD", "images"), base_filename_with_ext)
    cv2.imwrite(out_image_path, image_processed)

    # annotation process and save
    if row['tumor'] == 1 and os.path.exists(raw_json_path):
        with open(raw_json_path, 'r') as f:
            data = json.load(f)

        #Calculate the scale and padding
        scale = min((224, 224)[1] / original_w, (224, 224)[0] / original_h)
        new_width, new_height = int(original_w * scale), int(original_h * scale)
        top_pad = ((224, 224)[0] - new_height) // 2
        left_pad = ((224, 224)[1] - new_width) // 2

        # Update the coordinates for each point in each shape
        for shape in data['shapes']:
            new_points = []
            for x, y in shape['points']:
                # Scale the point and then add the padding offset
                new_x = x * scale + left_pad
                new_y = y * scale + top_pad
                new_points.append([new_x, new_y])
            shape['points'] = new_points
            
        # Update the image path, height, and width fields in the JSON
        data['imagePath'] = base_filename_with_ext
        data['imageHeight'] = (224, 224)[0]
        data['imageWidth'] = (224, 224)[1]
        data['imageData'] = None # Clear any embedded image data

        # Define the output path for the new JSON
        out_json_path = os.path.join(os.path.join("processed_dataset", "BTXRD", "Annotations"), json_filename)
            
        # Save the new, updated JSON file
        with open(out_json_path, 'w') as f:
            json.dump(data, f, indent=2)

# Copy the original excel file to the new directory for consistency
import shutil
shutil.copy(os.path.join(os.path.join("dataset", "BTXRD"), "dataset.xlsx"), os.path.join("processed_dataset", "BTXRD", "dataset.xlsx"))
print(f"Updated dataset created.")
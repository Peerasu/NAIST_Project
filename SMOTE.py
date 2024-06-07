import os
import cv2
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# Define directories
original_img_dir = '../../../../../mnt/d/peerasu/New/Patch_Train'
resampled_img_dir = '../../../../../mnt/d/peerasu/New/SMOTE_Patch_Train'
os.makedirs(resampled_img_dir, exist_ok=True)

# Load annotations
annot_train_file = pd.read_csv('../annotation_new/Label_Train.csv')

# Apply SMOTE to each image individually
for i, row in annot_train_file.iterrows():
    # Load original image
    img_name = row['Sample_Name']
    original_img_path = os.path.join(original_img_dir, img_name + '.png')
    original_img = cv2.imread(original_img_path)

    # Extract label
    label = row['Label']

    # Flatten image
    flattened_img = original_img.flatten()

    # Apply SMOTE to the flattened image
    smote = SMOTE(random_state=42)
    img_resampled, label_resampled = smote.fit_resample(flattened_img.reshape(-1, 1), [label])
    img_resampled = img_resampled.flatten()

    # Reshape the resampled image back to its original dimensions
    img_resampled = img_resampled.reshape(original_img.shape)

    # Save resampled image
    resampled_img_name = f'{row["Sample_Name"]}_resample_{i}'
    resampled_img_path = os.path.join(resampled_img_dir, resampled_img_name + '.png')
    cv2.imwrite(resampled_img_path, img_resampled)

    # Append resampled image info to DataFrame
    resampled_row = {'Sample_Name': resampled_img_name, 'Label': label_resampled[0]}
    annot_train_file = annot_train_file.append(resampled_row, ignore_index=True)

# Save updated annotations to a new CSV file
resampled_annotations_path = '../annotation_new/SMOTE_Label_Train.csv'
annot_train_file.to_csv(resampled_annotations_path, index=False)

# Now you have resampled images and annotations ready for training

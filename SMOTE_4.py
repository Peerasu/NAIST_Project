import os
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import shutil
import numpy as np
import cv2

# Define directories
original_img_dir = '../../../../../mnt/d/peerasu/New/Patch_Train'
resampled_img_dir = '../../../../../mnt/d/peerasu/New/SMOTE_Patch_Train'
os.makedirs(resampled_img_dir, exist_ok=True)

# Load annotations
annot_train_file = pd.read_csv('../annotation_new/Label_Train.csv')

# Extract image paths and labels
img_names = annot_train_file['Sample_Name'].values
img_names = [os.path.join(original_img_dir, filename + '.png') for filename in img_names]
labels = annot_train_file['Label'].values

# Initialize label encoder (if needed for your labels)
# label_encoder = LabelEncoder()
# Use label_encoder.fit(labels) on labels if your labels are strings

# Apply SMOTE
smote = SMOTE(random_state=42)

# Convert img_names to a 2D array with one column
img_names_array = img_names.reshape(-1, 1)

# Load and preprocess images
images = []
for img_path in img_names_array:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Example: grayscale
    img = img.astype('float32') / 255.0  # Normalize pixel values (0-1)
    images.append(img)
images_np = np.array(images)

# Apply SMOTE
smote = SMOTE()
img_paths_resampled_encoded, labels_resampled = smote.fit_resample(images_np, labels)

# Decode resampled image paths
img_paths_resampled = img_paths_resampled_encoded.flatten()

# Create a DataFrame with the resampled image paths and labels
resampled_annotations = pd.DataFrame({'Sample_Name': img_paths_resampled, 'Label': labels_resampled})

# Save the resampled annotations to a new CSV file
resampled_annotations.to_csv('../annotation_new/SMOTE_Label_Train.csv', index=False)

# Copy original images to resampled directory (not needed)
# This section was commented out as it's redundant since you already have resampled paths

# Copy resampled images to resampled directory with unique names
for idx, img_path_resampled in tqdm(enumerate(img_paths_resampled)):
  img_name_resampled, _ = os.path.splitext(os.path.basename(img_path_resampled))  # Extract filename without extension
  new_img_name = f"{img_name_resampled}_{idx}.png"  # Add index for uniqueness
  resampled_img_path = os.path.join(resampled_img_dir, new_img_name)
  shutil.copyfile(img_path_resampled, resampled_img_path)

# Now you have resampled images and annotations ready for training

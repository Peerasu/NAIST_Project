import os
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import shutil

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

# # Initialize label encoder
# label_encoder = LabelEncoder()

# Apply SMOTE
smote = SMOTE(random_state=42)
img_paths_resampled_encoded, labels_resampled = smote.fit_resample(img_names.reshape(-1, 1), labels)

# Decode resampled image paths
img_paths_resampled = img_paths_resampled_encoded.flatten()

# Create a DataFrame with the resampled image paths and labels
resampled_annotations = pd.DataFrame({'Sample_Name': img_paths_resampled, 'Label': labels_resampled})

# Save the resampled annotations to a new CSV file
resampled_annotations.to_csv('../annotation_new/SMOTE_Label_Train.csv', index=False)

# Copy original images to resampled directory
for img_path in tqdm(img_names):
    img_name = os.path.basename(img_path)
    resampled_img_path = os.path.join(resampled_img_dir, img_name)
    shutil.copyfile(img_path, resampled_img_path)

# Copy resampled images to resampled directory with unique names
for img_path_resampled in tqdm(img_paths_resampled):
    img_name_resampled = os.path.basename(img_path_resampled)
    resampled_img_path = os.path.join(resampled_img_dir, img_name_resampled)
    shutil.copyfile(img_path_resampled, resampled_img_path)

# Now you have resampled images and annotations ready for training

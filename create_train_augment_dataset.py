import os
import cv2
import numpy as np
import csv
from augmentation import *

# Define your train_augment function here
def train_augment(image):
    """
    Performs the first set of data augmentation on the image during training.
    Returns the augmented image.
    """
    image = do_random_flip(image)
    image = do_random_rot90(image)
    for fn in np.random.choice([
        lambda image: (image),
        lambda image: do_random_noise(image, mag=0.1),
        lambda image: do_random_contrast(image, mag=0.25)
    ], 1): image = fn(image)
    
    for fn in np.random.choice([
        lambda image: (image),
        lambda image: do_random_rotate_scale(image, angle=45, scale=[1, 2]),
    ], 1): image = fn(image)

    return image

# Path to your original dataset directory
original_dataset_dir = '../../../../../mnt/d/peerasu/New/Patch_Train'

# Path to the new dataset directory where augmented images will be saved
new_dataset_dir = '../../../../../mnt/d/peerasu/New/Patch_Train_BL'

# Path to the true train label CSV file
original_train_label = "../annotation_new/Label_Train.csv"

# Path to save the CSV file
csv_file_path = "../annotation_new/Label_Train_BL.csv"

# Create the new dataset directory if it doesn't exist
if not os.path.exists(new_dataset_dir):
    os.makedirs(new_dataset_dir)

# Load true train labels from CSV
true_train_labels = {}
with open(original_train_label, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        true_train_labels[row['Sample_Name']] = row['Label']

# Initialize list to store data for CSV
csv_data = []

# Iterate through each image in the original dataset
for filename in os.listdir(original_dataset_dir):
    # Load the original image
    original_image_path = os.path.join(original_dataset_dir, filename)
    original_image = cv2.imread(original_image_path)
    
    # Save the original image and its augmented versions into the new dataset directory
    cv2.imwrite(os.path.join(new_dataset_dir, filename), original_image)
    
    # Get label for this image from true_train_labels
    label = int(true_train_labels.get((filename.split('.'))[0]))
    
    # Append data to CSV list
    csv_data.append({'Sample_Name': (filename.split('.'))[0], 'Label': label})
    
    # Apply augmentation generate three augmented images
    if label == 1:
        augmented_images = []
        for _ in range(3):
            augmented_image = train_augment(original_image.copy())  # Make a copy to avoid modifying the original image
            augmented_images.append(augmented_image)
        
        for i, augmented_image in enumerate(augmented_images):
            augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}"
            cv2.imwrite(os.path.join(new_dataset_dir, augmented_filename + '.' + 'png'), augmented_image)
            
            # Append data to CSV list
            csv_data.append({'Sample_Name': augmented_filename, 'Label': label})

# Write data to CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Sample_Name', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for data in csv_data:
        writer.writerow(data)
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import os
import numpy as np
import pandas as pd
from resnet import *
from efficientnet_pytorch import EfficientNet
import math
from torchvision import models


# Load the pre-trained EfficientNet-B0 model
# model = EfficientNet.from_pretrained('efficientnet-b0')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Modify the final fully connected layer to match the number of classes (e.g., 3 classes)
num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load our own good model
model_path = '../../../../../mnt/d/peerasu/New/Models_BL_ResNet/model_1_epoch_1.pth'
model.load_state_dict(torch.load(model_path))

# Move model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = model.to(device)

# Define the transform (must match the training transform)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to load an image and apply transformations
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def Resize_Original_Image(name, data_path):
        mag = int((name.split('_')[2]).split('HE')[-1]) 
        factor = int(40 / mag)
        new_image = cv2.imread(os.path.join(data_path, name + '.' + 'tif'))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        new_image = cv2.resize(new_image, dsize=(448*factor, 448*factor), interpolation=cv2.INTER_LINEAR)
        
        return new_image, mag
    
def create_40_mag_image(image, mag):
    image_shape = image.shape
    factor = int(math.ceil(40 / mag))
    h_length = int(math.ceil(image_shape[0] / factor))
    w_length = int(math.ceil(image_shape[1] / factor))
    image = cv2.resize(image, dsize=(w_length*factor, h_length*factor), interpolation=cv2.INTER_LINEAR)
    # Image_Show(image)
    
    tile_list = []
    # tile_list_raw = []
    
    for i in range(factor):
        for j in range(factor):
            new_image = image[i*h_length:(i+1)*h_length, j*w_length:(j+1)*w_length]
            
            # Image_Show(new_image)
            
            # tile_list_raw.append(new_image)
            
            new_image = cv2.resize(new_image, dsize=(448, 448), interpolation=cv2.INTER_LINEAR)
            tile_list.append(new_image)
    return tile_list
    
# def do_thresholding(img):
#     grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_c = 255 - grayscale_img
#     thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thres, thres_img, img_c


def do_image_cut(tile, patch_size, Cut=1):
    # Thres_,_,_ = do_thresholding(original_image)
    patch_size = int(patch_size)
    # Image_Thres = (Thres_ - 22) * patch_size * patch_size * 3
    tile_shape = tile.shape
    h_num = int(math.ceil(tile_shape[0] / patch_size))
    w_num = int(math.ceil(tile_shape[1] / patch_size))
    # print(w_num, h_num)
    tile = cv2.resize(tile, dsize=(w_num*patch_size, h_num*patch_size), interpolation=cv2.INTER_LINEAR)
    # Block = np.zeros((tile_size, tile_size, image.shape[2]), dtype=np.uint8)
      
    patch_list = []
    # bg_list = []
    if Cut == 0:
        for i in range(h_num):
            for j in range(w_num):
                patch = tile[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                # background = 0
                # patch = torch.tensor(patch, dtype=torch.float32, device=device)
                patch_list.append(patch)
                # bg_list.append(background)
    else:
        for i in range(h_num):
            for j in range(w_num):
                patch = tile[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                # # otsu thresholding
                # patch_iN = 255 - patch
                # Patch_Thres = np.sum(patch_iN)
                
                # if Patch_Thres > Image_Thres:
                #     background = 0
                # else:
                #     background = 1

                # patch = torch.tensor(patch, dtype=torch.float32, device=device)
                patch_list.append(patch)
                # bg_list.append(background)
    return patch_list

# Function to perform inference on a single patch
def predict_patch(patch, model):
    patch = transform(patch).unsqueeze(0)  # Add batch dimension
    patch = patch.to(device)
    with torch.no_grad():
        outputs = model(patch)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Function to perform inference on a full image by aggregating patch predictions
def predict_image(name, image_path, model, patch_size=224):
    image, mag = Resize_Original_Image(name, image_path)
    tile_list = create_40_mag_image(image, mag)
    patch_list = []
    for tile in tile_list:
        patch_list_from_tile = do_image_cut(tile, patch_size)
        patch_list.extend(patch_list_from_tile)
        
    patch_predictions = [predict_patch(patch, model) for patch in patch_list]
    
    count = [0, 0, 0]    
    for pred in patch_predictions:
        if pred == 0:
            count[0] += 1
        elif pred == 1:
            count[1] += 1
        else:
            count[2] += 1
    print(f'Survive: {count[0]}, Death: {count[1]}, BG: {count[2]}')
        
    patch_predictions = [pred for pred in patch_predictions if pred in [0, 1]]
    aggregated_prediction = np.bincount(patch_predictions).argmax()  # Majority voting
    
    if count[0] == count[1]:
        aggregated_prediction = 1
        
    return aggregated_prediction, count

# Example usage on a directory of images
val_images_dir = '../../../../../mnt/d/peerasu/Image'
val_labels_csv = '../annotation_new/Annot_Val_Patient_WSI.csv'

# Load the test labels
val_labels = pd.read_csv(val_labels_csv)
image_names = [img for img in val_labels['Sample_Name']]

# Evaluate the model on the test set
correct = 0
total = 0
print(f'Total: {len(image_names)}')

# patient_predict = {}
conf = [0, 0, 0, 0]
for i, name in enumerate(image_names):
    # id = (name.split('_'))[0]
    # patient_predict[id] = []
    true_label = val_labels.iloc[i]['Death']
    predicted_label, count = predict_image(name, val_images_dir, model)
    # patient_predict[id].append(count)
    total += 1
    if predicted_label == true_label:
        correct += 1
    print(f'GT: {true_label}, Predict: {predicted_label}')
    
    if predicted_label == 0 and true_label == 1:
        conf[0] += 1
    elif predicted_label == 1 and true_label == 1:
        conf[1] += 1
    elif predicted_label == 1 and true_label == 0:
        conf[2] += 1
    elif predicted_label == 0 and true_label == 0:
        conf[3] += 1
    
    if total % 10 == 0 or total == len(image_names) - 1:
        print(f'Correct: {correct} / {total}')

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

print(conf)




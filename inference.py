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
import matplotlib.pyplot as plt


# Load the pre-trained EfficientNet-B0 model
# model = EfficientNet.from_pretrained('efficientnet-b0')
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)





# Load the pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Number of classes for the final output
num_classes = 3

# Define the custom classifier with additional fully connected layers
class CustomClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)  # First additional fully connected layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)          # Second additional fully connected layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)  # Final output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Replace the original fully connected layer with the custom classifier
in_features = model.fc.in_features
model.fc = CustomClassifier(in_features, num_classes)




# # Modify the final fully connected layer to match the number of classes (e.g., 3 classes)
# num_classes = 3
# model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load our own good model
model_path = '../../../../../mnt/d/peerasu/NewNew/Models_SS_Last_2/model_1_epoch_40.pth'
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
    return patch_list, w_num, h_num

def compute_class_weights(frequencies):
    total = sum(frequencies)
    weights = [total / f for f in frequencies]
    # Normalize weights to sum to 1
    weights = [w / sum(weights) for w in weights]
    return weights


def Image_Show(image):
    # b, g, r = cv2.split(image)
    # image = cv2.merge([r, g, b])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image / 255)
    # plt.axis('off')
    plt.show()
    
def Add_Border(image_list,Bold_Image_Size, bg_list):
    New_List_Image=[]
    for i, image in enumerate(image_list):
        if bg_list[i] == 0:
            img = cv2.copyMakeBorder(image, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                            cv2.BORDER_CONSTANT,
                                            value=(0, 255, 0))
        elif bg_list[i] == 1:
            img = cv2.copyMakeBorder(image, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                            cv2.BORDER_CONSTANT,
                                            value=(255, 0, 0))
        else:
            img = cv2.copyMakeBorder(image, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                            cv2.BORDER_CONSTANT,
                                            value=(0, 0, 255))
        New_List_Image.append(img)
    return New_List_Image

def Tile_Image(tile_list, image_shape, tile_size):
    # w_num = math.ceil(int(image_shape[0])/int(image_size))
    # h_num = math.ceil(int(image_shape[1])/int(image_size))
    # print(len(image_list))
    #
    h_num = int(image_shape[0])
    w_num = int(image_shape[1])
    image_shapes0 = int(h_num*tile_size)
    image_shapes1 = int(w_num*tile_size)
    image = np.zeros((image_shapes0, image_shapes1, 3), dtype=float, order='C')
    
    index = 0
    for i in range(h_num):
        for j in range(w_num):
            tile_image = tile_list[index]
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile_image
            index+=1
    return image


def Add_Border_Show(image_list,Bold_Image_Size):
    New_List_Image=[]
    for i, image in enumerate(image_list):
        img = cv2.copyMakeBorder(image, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                        cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
        New_List_Image.append(img)
    return New_List_Image



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
    
    print(name)
    Image_Show(image)
    
    patch_list = []
    for tile in tile_list:
        patch_list_from_tile, w_num, h_num = do_image_cut(tile, patch_size)
        patch_list.extend(patch_list_from_tile)
    patch_predictions = [predict_patch(patch, model) for patch in patch_list]
    
    border_size = 4
    patch_size = 224
    Image_Tile_list = []
    # SHOW FULL TILE FROM PATCH
    Border_List = Add_Border(patch_list, border_size, patch_predictions)
    new_patch_size = patch_size + border_size * 2
    image_shape = [w_num, h_num]
    Image_Tile = Tile_Image(Border_List, image_shape, new_patch_size)
    Image_Tile_list.append(Image_Tile)
        
    # num_square = int(math.sqrt(len(tile_list)))
    # Border_List = Add_Border_Show(Image_Tile_list, border_size*2)
    # new_patch_size = new_patch_size*w_num + border_size * 4
    # image_shape = [num_square, num_square]
    # Image_Tile = Tile_Image(Border_List, image_shape, new_patch_size)
    Image_Show(Image_Tile)
    
    
    count = [0, 0, 0]    
    for pred in patch_predictions:
        if pred == 0:
            count[0] += 1
        elif pred == 1:
            count[1] += 1
        else:
            count[2] += 1
    print(f'Survive: {count[0]}, Death: {count[1]}, BG: {count[2]}')
        
    # patch_predictions = [pred for pred in patch_predictions if pred in [0, 1]]
    # aggregated_prediction = np.bincount(patch_predictions).argmax()  # Majority voting
    
    # # if count[0] == count[1]:
    # #     aggregated_prediction = 1
        
    # return aggregated_prediction, count
    
    freq = [1872, 1749, 1690]       # freq from train dataset
    weight = compute_class_weights(freq)
    final_predict_weight = [count[0]*weight[0], count[1]*weight[1]]
    
    if final_predict_weight[0] > final_predict_weight[1]:
        final_predict = 0
    else:
        final_predict = 1
    
    return final_predict, count

# Example usage on a directory of images
val_images_dir = '../../../../../mnt/d/peerasu/Image'
val_labels_csv = '../annotation_newnew/Label_Test_WSI.csv'

# Load the test labels
val_labels = pd.read_csv(val_labels_csv)

image_names = []
id_list = {}
for i, row in val_labels.iterrows():
    name = row['Sample_Name']
    label = row['Death']
    
    image_names.append(name)
    id = name.split('_')[0]
    if id not in id_list.keys():
        id_list[id] = label
    
    
    

# Evaluate the model on the test set
correct = 0
total = 0
print(f'Total: {len(image_names)}')

# patient_predict = {}
conf = [0, 0, 0, 0]
# patients = {}
for i, name in enumerate(image_names):
    id = (name.split('_'))[0]
    true_label = val_labels.iloc[i]['Death']
    predicted_label, count = predict_image(name, val_images_dir, model)
    
    # if id not in patients:
    #     patients[id] = []
    # patients[id].append(predicted_label)
    
    
    if predicted_label == 1 and true_label == 1:
        conf[0] += 1
    elif predicted_label == 0 and true_label == 1:
        conf[1] += 1
    elif predicted_label == 1 and true_label == 0:
        conf[2] += 1
    elif predicted_label == 0 and true_label == 0:
        conf[3] += 1
    
    
    # patient_predict[id].append(count)
    total += 1
    if predicted_label == true_label:
        correct += 1
    print(f'GT: {true_label}, Predict: {predicted_label}')
    
    # if total % 10 == 0 or total == len(image_names) - 1:
    #     print(f'Correct: {correct} / {total}')

# accuracy = 100 * correct / total

# conf = [0, 0, 0, 0]
# correct = 0
# total = len(id_list.keys())
# for id in list(patients.keys()):
#     survive = 0
#     death = 0
#     for pred in patients[id]:
#         if pred == 0:
#             survive += 1
#         else:
#             death += 1
            
#     if survive > death:
#         final_pred = 0
#     else:
#         final_pred = 1
    
#     if final_pred == id_list[id]:
#         correct += 1
        
#     if final_pred == 1 and id_list[id] == 1:
#         conf[0] += 1
#     elif final_pred == 0 and id_list[id] == 1:
#         conf[1] += 1
#     elif final_pred == 1 and id_list[id] == 0:
#         conf[2] += 1
#     elif final_pred == 0 and id_list[id] == 0:
#         conf[3] += 1
        
# patients_dist = [0, 0]
# for id in id_list:
#     if id_list[id] == 0:
#         patients_dist[0] += 1
#     else:
#         patients_dist[1] += 1
# print(f"Total Survive: {patients_dist[0]}, Total Death: {patients_dist[1]}")
        
        
        
accuracy = correct / total * 100
print(f'Validation Accuracy: {accuracy:.2f}%')
print(conf)




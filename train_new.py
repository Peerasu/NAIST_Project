import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet
from augmentation import train_augment
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_class_weights(frequencies):
    total = sum(frequencies)
    weights = [total / f for f in frequencies]
    weights = [w / sum(weights) for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)

def Image_Show(image):
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image / 255)
    plt.show()

# Custom Dataset class with pre-loaded data and augmentations
class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, img_dir, transform=None, do_augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.do_augment = do_augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_paths[idx] + '.png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.do_augment:
            image = train_augment(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Compute classes distribution
annot_train_file = pd.read_csv('../annotation_new/Patch_Annot_Train.csv')
dis = [0, 0, 0]
for _, raw in annot_train_file.iterrows():
    if raw['Background'] == 0:
        if raw['Death'] == 0:
            dis[0] += 1
        else:
            dis[1] += 1
    else:
        dis[2] += 1

# Class frequencies
class_frequencies = [dis[0], dis[1], dis[2]]
print("Class Frequencies: ", class_frequencies)

# Extract image paths and labels
img_paths = annot_train_file['Sample_Name'].values
labels = annot_train_file['Label'].values

# Reshape for SMOTE
img_paths = img_paths.reshape(-1, 1)

# Apply SMOTE
smote = SMOTE(random_state=42)
img_paths_resampled, labels_resampled = smote.fit_resample(img_paths, labels)

# Flatten the image paths array
img_paths_resampled = img_paths_resampled.flatten()

# Create the dataset
train_dataset = CustomDataset(img_paths=img_paths_resampled, labels=labels_resampled, img_dir='../../../../../mnt/d/peerasu/New/Patch_Train_SMOTE', transform=transform, do_augment=True)
test_dataset = CustomDataset(img_paths=pd.read_csv('../annotation_new/Label_Test.csv')['Sample_Name'].values, labels=pd.read_csv('../annotation_new/Label_Test.csv')['Label'].values, img_dir='../../../../../mnt/d/peerasu/New/Patch_test', transform=transform, do_augment=False)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the pre-trained EfficientNet-B0 model
model = EfficientNet.from_pretrained('efficientnet-b0')
num_classes = 3
model._fc = nn.Linear(model._fc.in_features, num_classes)
model = model.to(device)

# Compute class weights and define the loss function
class_weights = compute_class_weights(class_frequencies)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training and evaluation loop
num_epochs = 100

# Create directory for saving models if it doesn't exist
model_save_path = '../../../../../mnt/d/peerasu/New/Models_WCE'
result_save_path = '../Train_Results_WCE'
model_num = 1
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    print(f'####### Epoch {epoch+1}/{num_epochs} #######')

    for i, (inputs, labels) in enumerate(tqdm(train_loader, unit='batch')):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Print progress
        if i % 100 == 0:  # Adjust the frequency of printing if needed
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, Learning Rate: {current_lr}')

    epoch_loss = running_loss / len(train_loader.dataset)

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Step the scheduler
    scheduler.step()

    # Evaluation on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * correct / total

    # Save epoch results to CSV
    epoch_results = {
        'model_num': [model_num],
        'epoch': [epoch + 1],
        'learning_rate': [current_lr],
        'train_loss': [epoch_loss],
        'test_loss': [test_loss],
        'test_accuracy': [test_accuracy]
    }

    df = pd.DataFrame(epoch_results)
    result_file_path = f'{result_save_path}/result_model_{model_num}.csv'

    # If the file already exists, append data; otherwise, create a new file
    if os.path.isfile(result_file_path):
        with open(result_file_path, 'a') as f:
            pd.DataFrame(epoch_results).to_csv(f, header=False, index=False)
    else:
        pd.DataFrame(epoch_results).to_csv(result_file_path, header=True, index=False)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{model_num}_epoch_{epoch+1}.pth'))

    print(f'------- Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Learning Rate: {current_lr} -------')

print('Training complete')

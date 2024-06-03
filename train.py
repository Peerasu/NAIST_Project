import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm  # For progress bar
from resnet import *
from augmentation import *

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + '.' + 'png')
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
train_dataset = CustomDataset(csv_file='../annotation/true_train_label_4_augment.csv', img_dir='../../../../../mnt/d/peerasu/Tile_train_4_augment', transform=train_transform)
test_dataset = CustomDataset(csv_file='../annotation/true_test_label_4.csv', img_dir='../../../../../mnt/d/peerasu/Tile_test_4', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize model
model = get_resnet("ResNet50")

# Load the good model
model_path = '../../../../../mnt/d/peerasu/models/model_1_epoch_9.pth'
model.load_state_dict(torch.load(model_path))


# # Load the pre-trained ResNet50 model from torchvision
# model = models.resnet50(pretrained=True)

# # Modify the final fully connected layer to match the number of classes (e.g., 3 classes)
# num_classes = 3
# model.fc = nn.Linear(model.fc.in_features, num_classes)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training and evaluation loop
num_epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create directory for saving models if it doesn't exist
model_save_path = '../../../../../mnt/d/peerasu/models_from_model_1_epoch_9'
result_save_path = '../train_results_from_model_1_epoch_9'
# model_save_path = '../../../../../mnt/d/peerasu/models_2'
# result_save_path = '../train_results'
model_num = 1
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
    
    
    
# # # Visualize Model
    
# import torch
# import torch.nn as nn
# from torchsummary import summary
# from torchviz import make_dot

# # Assuming the ResNet and get_resnet definitions are already provided

# # Print the model architecture
# print(model)

# # Alternatively, use torchsummary to get a detailed summary
# summary(model, input_size=(3, 224, 224))

# # Create a sample input tensor
# sample_input = torch.randn(1, 3, 224, 224).to(device)

# # Pass the sample input through the model
# output = model(sample_input)

# # Generate a visualization of the model
# dot = make_dot(output, params=dict(model.named_parameters()))
# dot.render("../resnet50_architecture", format="png")
    
    

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    print(f'####### Epoch {epoch+10}/{num_epochs} #######')
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
        'epoch': [epoch + 10],
        'learning_rate': [current_lr],
        'train_loss': [epoch_loss],
        'test_loss': [test_loss],
        'test_accuracy': [test_accuracy]
    }

    
    df = pd.DataFrame(epoch_results)
    result_file_path = f'{result_save_path}/result_model_{model_num}_from_model_1_epoch_9.csv'

    # If the file already exists, append data; otherwise, create a new file
    if os.path.isfile(result_file_path):
        with open(result_file_path, 'a') as f:
            pd.DataFrame(epoch_results).to_csv(f, header=False, index=False)
    else:
        pd.DataFrame(epoch_results).to_csv(result_file_path, header=True, index=False)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{model_num}_epoch_{epoch+10}.pth'))
        
    print(f'------- Epoch {epoch+10}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Learning Rate: {current_lr} -------')

print('Training complete')
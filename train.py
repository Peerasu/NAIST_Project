import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import os
from tqdm import tqdm  # For progress bar
from efficientnet_pytorch import EfficientNet
from augmentation import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Image_Show(image):
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image / 255)
    # plt.axis('off')
    plt.show()

# Custom Dataset class with pre-loaded data and augmentations
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, do_augment=False):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.do_augment = do_augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + '.' + 'png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels.iloc[idx, 1]
        
        if self.do_augment == True:
            image = train_augment(image)
            # Image_Show(image)

        if self.transform:
            image = self.transform(image)

        return image, label
    

def train_augment(image):
    """
    Performs the first set of data augmentation on the image during training.
    Returns the augmented image.
    """
    
    random_flip_choices = [
        lambda image: do_random_flip(image),
        lambda image: do_random_rot90(image)
    ]
    
    rotate_scale_choices = [
        lambda image: (image),
        lambda image: do_random_rotate_scale(image, angle=45, scale=[0.8, 2])
    ]
    
    augmentation_choices = [
        lambda image: (image),
        lambda image: do_random_noise(image, mag=0.1),
        lambda image: do_random_contrast(image, mag=0.25)
    ]
    
    if np.random.rand() < 0.5:
        image = np.random.choice(random_flip_choices)(image)
        if np.random.rand() < 0.8:
            image = np.random.choice(rotate_scale_choices)(image)
        if np.random.rand() < 0.5:
            image = np.random.choice(augmentation_choices)(image)

    return image


# Define transformation for training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Dataset and DataLoader
train_dataset = CustomDataset(csv_file='../annotation_new/Label_Train.csv', img_dir='../../../../../mnt/d/peerasu/New/Patch_Train', transform=transform, do_augment=True)
test_dataset = CustomDataset(csv_file='../annotation_new/Label_Test.csv', img_dir='../../../../../mnt/d/peerasu/New/Patch_test', transform=transform, do_augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the pre-trained EfficientNet-B0 model
model = EfficientNet.from_pretrained('efficientnet-b0')

# Modify the final fully connected layer to match the number of classes (e.g., 3 classes)
num_classes = 3
model._fc = nn.Linear(model._fc.in_features, num_classes)

# Load our own good model
model_path = '../../../../../mnt/d/peerasu/New/Models/model_1_epoch_10.pth'
model.load_state_dict(torch.load(model_path))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the element-wise cross-entropy loss
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        # Compute the probability of the true class
        pt = torch.exp(-BCE_loss)
        
        if self.alpha is not None:
            # Get the alpha value for each target
            at = self.alpha.gather(0, targets)
            # Apply alpha to the loss
            F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def compute_class_weights(frequencies):
    total = sum(frequencies)
    weights = [total / f for f in frequencies]
    # Normalize weights to sum to 1
    weights = [w / sum(weights) for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)
        
# Compute classes distribution
annot_train_file = pd.read_csv('../annotation_new/Patch_Annot_Train.csv')
dis = [0, 0, 0]
for i, raw in annot_train_file.iterrows():
    if raw['Background'] == 0:
        if raw['Death'] == 0:
            dis[0] += 1
        else:
            dis[1] += 1
    else:
        dis[2] += 1
        
# Class frequencies
class_frequencies = [dis[0], dis[1], dis[2]]
class_weights = compute_class_weights(class_frequencies)

# Loss function and optimizer
criterion = FocalLoss(alpha=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training and evaluation loop
num_epochs = 100

model = model.to(device)

# Create directory for saving models if it doesn't exist
model_save_path = '../../../../../mnt/d/peerasu/New/Models'
result_save_path = '../Train_Results'
model_num = 1
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    print(f'####### Epoch {epoch+11}/{num_epochs} #######')

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
        'epoch': [epoch + 11],
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
    torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{model_num}_epoch_{epoch+11}.pth'))

    print(f'------- Epoch {epoch+11}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Learning Rate: {current_lr} -------')

print('Training complete')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import pickle
import cv2
from os import listdir
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = (256, 256)
width, height, depth = 256, 256, 3

import os
import torch
from torchvision import transforms
from PIL import Image

# Function to convert image to array
def convert_image_to_array(image_path):
    try:
        image = Image.open(image_path)
        if image is not None:
            image = image.resize(default_image_size)
            return transforms.ToTensor()(image)
        else:
            return torch.tensor([])
    except Exception as e:
        print(f"Error: {e}")
        return None

# Initialize variables
image_list, label_list = [], []
directory_root = '/home/suhrid/EE585 Project/EE585_Project_Dataset'
default_image_size = (256, 256)

# Loop through directories
for plant_folder in os.listdir(directory_root):
    plant_folder_path = os.path.join(directory_root, plant_folder)
    if not os.path.isdir(plant_folder_path):
        continue

    # Loop through images in each folder
    for image_name in os.listdir(plant_folder_path)[:200]:
        image_path = os.path.join(plant_folder_path, image_name)
        if not (image_name.endswith(".jpg") or image_name.endswith(".JPG")):
            continue

        # Convert image to array and append to lists
        image_array = convert_image_to_array(image_path)
        if image_array is not None:
            image_list.append(image_array)
            label_list.append(plant_folder)

# Convert image and label lists to tensors
image_data = torch.stack(image_list)
# image_labels = torch.tensor(label_list)

# Print the number of images loaded
print(f"Number of images loaded: {len(image_list)}")

#1-hot encoding the classes
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)

image_data = np.array(image_data, dtype=np.float16) / 225.0

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(image_data, image_labels, test_size=0.2, random_state = 42)


# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDataset instances
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Define batch size
batch_size = 32

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





# Define transformations for data augmentation
aug = transforms.Compose([
    transforms.RandomRotation(25),                 # Random rotation by maximum 25 degrees
    transforms.RandomHorizontalFlip(),             # Random horizontal flip
    transforms.RandomVerticalFlip(),               # Random vertical flip
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2, scale=(0.8, 1.2)),  # Random affine transformation (shift, shear, scale)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),           # Random changes in color
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.75, 1.333)),           # Random resized crop
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),              # Random perspective transformation
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),  # Random erasing
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),             # Normalization
])


#________________________________________________________________________________________________________________________________________________________


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, num_classes)  # Output size modified to match [batch_size, 38]

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.batchnorm5 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(self.relu(self.batchnorm2(self.conv2(x))))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.dropout(x)

        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.pool(self.relu(self.batchnorm5(self.conv5(x))))
        x = self.dropout(x) #[32, 128, 8, 8]

        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



#________________________________________________________________________________________________________________________________________________________

# Create an instance of the model
model = ConvNet(num_classes=n_classes)

criterion = nn.CrossEntropyLoss()
#Setting the OPTIMIZER
#optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR / EPOCHS) #TODO: Change the optimizer to see the difference in results
#optimizer = optim.Adagrad(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR / EPOCHS)
#optimizer = optim.RMSprop(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR / EPOCHS)
#optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR / EPOCHS, amsgrad=True)
#optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=INIT_LR / EPOCHS)
optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9, nesterov=True, weight_decay=INIT_LR / EPOCHS)


#________________________________________________________________________________________________________________________________________________________

#Training and Testing the model

train_accuracy_list = []
test_accuracy_list = []

model.to(device)
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        # Apply data augmentation
        # inputs = torch.stack([aug(image) for image in inputs])
        optimizer.zero_grad()
        outputs = model(inputs)
        # Convert outputs to one-hot encoded labels
        # _, predicted = torch.max(outputs, 1)
        # one_hot_predicted = torch.eye(n_classes)[predicted]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        # one_hot_predicted = torch.eye(targets.shape[1]).to(device)[predicted]
        total += targets.size(0)
        class_labels = torch.argmax(targets, dim=1)
        correct += (predicted == class_labels).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    train_accuracy = correct / total
    train_accuracy_list.append(train_accuracy)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}')
    
# Plot the accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_accuracy_list, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy with SGD with Nesterov Momentum')
plt.legend()
plt.show()
plt.savefig('/home/suhrid/EE585 Project/Plots/Train_Accuracy_SGD_nesterov_Momentum')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        class_labels = torch.argmax(targets, dim=1)
        total += targets.size(0)
        correct += (predicted == class_labels).sum().item()
test_accuracy = correct / total
test_accuracy_list.append(test_accuracy)
print(f'Test Accuracy: {test_accuracy:.4f}')

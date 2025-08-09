# To-Do
# Set configuration : like EPOCHS, BATCH_SIZE, learning rate, GPU etc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from methodology.dual_branch_test.model import DualBranchModel
from dataset import ViolenceDataset
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class EarlyStopper:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0

    def check(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded

# torch.cuda.empty_cache()
""" Configuration """
EPOCHS = 10
BATCH_SIZE = 1
LR = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

""" Transform (optional but recommended) """
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

""" Paths """
CSV_BASE_PATH = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\Data split"
NPY_ROOT = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\npy_segments"

train_csv = os.path.join(CSV_BASE_PATH, 'train.csv')
val_csv = os.path.join(CSV_BASE_PATH, 'val.csv')

""" Datasets & Loaders """
train_dataset = ViolenceDataset(train_csv, NPY_ROOT, balance=True)
val_dataset = ViolenceDataset(val_csv, NPY_ROOT, balance=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

""" Model, Loss, Optimiser """
model = DualBranchModel(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
early_stopper = EarlyStopper(patience=3)

""" Training Loop """

train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch')

    for inputs, labels in progress:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        acc = correct / total * 100
        progress.set_postfix(loss=loss.item(), acc=f'{acc:.2f}%')
    
    # Validation 
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(DEVICE)
            val_labels = val_labels.to(DEVICE)

            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_labels)

            val_loss += loss.item()
            _, val_predicted = val_outputs.max(1)
            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = val_correct / val_total * 100
    train_accuracies.append(acc)
    val_accuracies.append(val_acc)

    if early_stopper.check(val_acc):
        print(f"Early stopping at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%")
        break

    print(f'[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    

torch.save(model.state_dict(), 'dual_branch_model.pth')
print("Model weights saved to 'dual_branch_model.pth'")

plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

np.save('train_accuracies.npy', np.array(train_accuracies))
np.save('val_accuracies.npy', np.array(val_accuracies))
print("Accuracies saved to 'train_accuracies.npy' and 'val_accuracies.npy'")












































# 1st try
# Training

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from model import DualBranchModel
# from dataset import ViolenceDataset
# from torchvision import transforms
# import os
# from tqdm import tqdm
# import numpy as np


# """ Configuration """
# EPOCHS = 10
# BATCH_SIZE = 1
# LR = 1e-4
# NUM_CLASSES = 2
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {DEVICE}')

# """ Transform (optional but recommended) """
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# """ Paths """
# CSV_BASE_PATH = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\Data split"
# FRAME_ROOT = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\Frames"

# train_csv = os.path.join(CSV_BASE_PATH, 'train.csv')
# val_csv = os.path.join(CSV_BASE_PATH, 'val.csv')

# """ Datasets & Loaders """
# train_dataset = ViolenceDataset(train_csv, FRAME_ROOT, transform=transform)
# val_dataset = ViolenceDataset(val_csv, FRAME_ROOT, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# """ Model, Loss, Optimiser """
# model = DualBranchModel(num_classes=NUM_CLASSES).to(DEVICE)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# """ Training Loop """

# train_accuracies = []
# val_accuracies = []

# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch')

#     for inputs, labels in progress:
#         inputs = inputs.to(DEVICE)
#         labels = labels.to(DEVICE)

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)

#         acc = correct / total * 100
#         progress.set_postfix(loss=loss.item(), acc=f'{acc:.2f}%')
    
#     # Validation 
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0

#     with torch.no_grad():
#         for val_inputs, val_labels in val_loader:
#             val_inputs = val_inputs.to(DEVICE)
#             val_labels = val_labels.to(DEVICE)

#             val_outputs = model(val_inputs)
#             loss = criterion(val_outputs, val_labels)

#             val_loss += loss.item()
#             _, val_predicted = val_outputs.max(1)
#             val_correct += (val_predicted == val_labels).sum().item()
#             val_total += val_labels.size(0)

#     val_acc = val_correct / val_total * 100
#     train_accuracies.append(acc)
#     val_accuracies.append(val_acc)

#     print(f'[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

# torch.save(model.state_dict(), 'dual_branch_model.pth')
# print("Model weights saved to 'dual_branch_model.pth'")

# np.save('train_accuracies.npy', np.array(train_accuracies))
# np.save('val_accuracies.npy', np.array(val_accuracies))
# print("Accuracies saved to 'train_accuracies.npy' and 'val_accuracies.npy'")
Estructura 
````
!mkdir -p /content/drive/MyDrive/gabor_project
!mkdir -p /content/drive/MyDrive/gabor_project/data
!mkdir -p /content/drive/MyDrive/gabor_project/models
!mkdir -p /content/drive/MyDrive/gabor_project/utils

!touch /content/drive/MyDrive/gabor_project/models/gabor_cnn.py
!touch /content/drive/MyDrive/gabor_project/utils/gabor_filters.py
!touch /content/drive/MyDrive/gabor_project/utils/dataloader.py
!touch /content/drive/MyDrive/gabor_project/train.py
!touch /content/drive/MyDrive/gabor_project/evaluate.py
````

DATASET

````
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
!unzip brain-tumor-mri-dataset.zip -d /content/data
````

UTILS 

dataloader
`````python
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.gabor_filters import build_gabor_kernels, apply_gabor


class GaborDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []
        self.labels = []
        self.kernels = build_gabor_kernels()

        classes = sorted(os.listdir(root_dir))

        for idx, cls in enumerate(classes):
            class_path = os.path.join(root_dir, cls)
            for img in os.listdir(class_path):
                self.paths.append(os.path.join(class_path, img))
                self.labels.append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.resize(img, (150,150))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gabor = apply_gabor(img, self.kernels)

        gabor = gabor / 255.0
        gabor = np.transpose(gabor, (2,0,1))  # CLAVE

        return torch.tensor(gabor, dtype=torch.float32), self.labels[idx]
``````

````python
from torch.utils.data import DataLoader
from utils.dataloader import GaborDataset

train_dataset = GaborDataset('/content/data/Training')
test_dataset = GaborDataset('/content/data/Testing')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
````

gabor_filters
````python
import cv2
import numpy as np

def build_gabor_kernels():
    kernels = []

    ksize = 21
    sigmas = [1]
    thetas = np.arange(0, np.pi, np.pi / 4)
    lambdas = [10]
    gammas = [0.5]

    for sigma in sigmas:
        for theta in thetas:
            for lambd in lambdas:
                for gamma in gammas:
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize),
                        sigma,
                        theta,
                        lambd,
                        gamma,
                        0,
                        ktype=cv2.CV_32F
                    )
                    kernels.append(kernel)

    return kernels


def apply_gabor(image, kernels):
    responses = []

    for kernel in kernels:
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        responses.append(filtered)

    return np.stack(responses, axis=-1)
````

MODELS 

gabor_cnn
````python
import torch
import torch.nn as nn


# =====================
# SE BLOCK
# =====================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        y = nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)

        return x * y


# =====================
# MODEL
# =====================
class GaborCNN(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()

        # Conv inicial
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # SE block (paper)
        self.se = SEBlock(128)

        self.pool = nn.MaxPool2d(2)

        # Residual block
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)

        # GAP
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # FC más profundo (como paper)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):

        # Conv inicial
        x = torch.relu(self.bn1(self.conv1(x)))

        # SE block
        x = self.se(x)

        x = self.pool(x)

        # Residual
        identity = x

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += identity
        x = torch.relu(x)

        x = self.dropout(x)

        # GAP
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # FC
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
````

````pyhton
import torch
import torch.nn as nn
from models.gabor_cnn import GaborCNN

model = GaborCNN(in_channels=32)  # depende de cuantos kernels generes

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
````
TRAIN

````python
import sys
sys.path.append('/content/drive/MyDrive/gabor_project')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gabor_cnn import GaborCNN
from utils.dataloader import GaborDataset

# =====================
# DATA
# =====================
train_dataset = GaborDataset('/content/data/Training')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# AUTO DETECT CHANNELS
# =====================
sample, _ = next(iter(train_loader))
in_channels = sample.shape[1]

# =====================
# MODEL
# =====================
model = GaborCNN(in_channels=in_channels).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =====================
# TRAIN
# =====================
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # progreso por batch
        if i % 20 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total

    print(f"\n Epoch {epoch+1}/{EPOCHS} COMPLETED")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%\n")

# =====================
# SAVE MODEL
# =====================
torch.save(model.state_dict(), '/content/drive/MyDrive/gabor_project/model.pth')
````

EVALUATE

````python
import sys
sys.path.append('/content/drive/MyDrive/gabor_project')

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from models.gabor_cnn import GaborCNN
from utils.dataloader import GaborDataset

# =====================
# DATA
# =====================
test_dataset = GaborDataset('/content/data/Testing')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# =====================
# MODEL
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# detectar canales automáticamente
sample, _ = next(iter(test_loader))
in_channels = sample.shape[1]

model = GaborCNN(in_channels=in_channels).to(device)

# cargar pesos
model.load_state_dict(torch.load('/content/drive/MyDrive/gabor_project/model.pth'))
model.eval()

# =====================
# EVALUACIÓN
# =====================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.float().to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =====================
# MÉTRICAS
# =====================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = np.mean(all_preds == all_labels)

print("\n==============================")
print("RESULTADOS DEL MODELO GABOR")
print("==============================")
print(f"\nAccuracy: {accuracy*100:.2f}%\n")

# nombres de clases (dataset brain tumor)
class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("Confusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))
````

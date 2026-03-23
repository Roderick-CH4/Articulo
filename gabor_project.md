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
        img = cv2.resize(img, (150, 150))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gabor = apply_gabor(img, self.kernels)

        # Normalización robusta
        gabor = gabor / (np.max(gabor) + 1e-6)

        # (H, W, C) → (C, H, W)
        gabor = np.transpose(gabor, (2, 0, 1))

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
import numpy as np
import cv2


# =========================
# GABOR KERNEL (PAPER)
# =========================
def gabor_kernel(ksize, sigma, theta, frequency):
    half = ksize // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]

    # Rotación
    x_theta = x * np.cos(theta) + y * np.sin(theta)

    # Gabor real e imaginario
    real = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * np.cos(2 * np.pi * frequency * x_theta)
    imag = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * np.sin(2 * np.pi * frequency * x_theta)

    return real.astype(np.float32), imag.astype(np.float32)


# =========================
# BUILD FILTER BANK (PAPER)
# =========================
def build_gabor_kernels():

    kernels = []

    # Frecuencias del paper
    f1 = 0.1
    f2 = 0.23
    f3 = 0.4

    # Orientaciones por escala (paper)
    orientations = [4, 6, 8]
    freqs = [f1, f2, f3]

    for f, num_theta in zip(freqs, orientations):

        thetas = np.linspace(0, np.pi, num_theta, endpoint=False)

        # sigma según paper (relación con frecuencia)
        sigma = 1 / (2 * np.pi * f)

        # tamaño kernel dinámico (paper)
        ksize = int(2 * 3 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        for theta in thetas:
            real, imag = gabor_kernel(ksize, sigma, theta, f)
            kernels.append((real, imag))

    return kernels


# =========================
# APPLY GABOR (PAPER)
# =========================
def apply_gabor(image, kernels):

    responses = []

    image = image.astype(np.float32)

    # Gabor complejo → magnitud
    for real_k, imag_k in kernels:

        real_resp = cv2.filter2D(image, cv2.CV_32F, real_k)
        imag_resp = cv2.filter2D(image, cv2.CV_32F, imag_k)

        magnitude = np.sqrt(real_resp**2 + imag_resp**2)
        responses.append(magnitude)

    # =====================
    # LPF (paper)
    # =====================
    lpf = cv2.GaussianBlur(image, (5, 5), 0)
    responses.append(lpf)

    # =====================
    # HPF (paper)
    # =====================
    hpf = cv2.Laplacian(image, cv2.CV_32F)
    responses.append(hpf)

    # Stack → (H, W, C)
    responses = np.stack(responses, axis=-1)

    # =====================
    # Anti-aliasing (paper)
    # =====================
    responses = cv2.GaussianBlur(responses, (3, 3), 0)

    # =====================
    # Downsampling (paper)
    # =====================
    responses = responses[::2, ::2, :]

    return responses
````

MODELS 

gabor_cnn
````python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()

        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)

        return x * y


class GaborCNN(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.se = SEBlock(128)

        self.pool = nn.MaxPool2d(2)

        # Residual
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.se(x)
        x = self.pool(x)

        identity = x

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += identity
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
````

````pyhton
import torch
import torch.nn as nn
from models.gabor_cnn import GaborCNN

sample, _ = next(iter(train_loader))
model = GaborCNN(in_channels=sample.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
````
TRAIN

````python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gabor_cnn import GaborCNN
from utils.dataloader import GaborDataset

train_dataset = GaborDataset('/content/data/Training')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample, _ = next(iter(train_loader))
in_channels = sample.shape[1]

model = GaborCNN(in_channels=in_channels).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if i % 20 == 0:
            print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {total_loss/len(train_loader):.4f}")
    print(f"Accuracy: {100*correct/total:.2f}%")
````

EVALUATE

````python
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from models.gabor_cnn import GaborCNN
from utils.dataloader import GaborDataset

test_dataset = GaborDataset('/content/data/Testing')
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample, _ = next(iter(test_loader))
in_channels = sample.shape[1]

model = GaborCNN(in_channels=in_channels).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/gabor_project/model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

print(f"\nAccuracy: {accuracy*100:.2f}%\n")

class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

print(classification_report(all_labels, all_preds, target_names=class_names))
print(confusion_matrix(all_labels, all_preds))
````

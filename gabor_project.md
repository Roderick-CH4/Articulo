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
!touch /content/drive/MyDrive/gabor_project/preprocesamiento.py
````

DATASET

````
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
!unzip brain-tumor-mri-dataset.zip -d /content/data
````

PREPROCESAMIENTO 
````python
import os
import cv2
import numpy as np
from tqdm import tqdm

from utils.gabor_filters import build_gabor_kernels, apply_gabor

INPUT_DIR = "/content/data"
OUTPUT_DIR = "/content/data_gabor"

os.makedirs(OUTPUT_DIR, exist_ok=True)

kernels = build_gabor_kernels()

for split in ["Training", "Testing"]:
    in_path = os.path.join(INPUT_DIR, split)
    out_path = os.path.join(OUTPUT_DIR, split)

    os.makedirs(out_path, exist_ok=True)

    classes = os.listdir(in_path)

    for cls in classes:
        os.makedirs(os.path.join(out_path, cls), exist_ok=True)

        for img_name in tqdm(os.listdir(os.path.join(in_path, cls)), desc=f"{split}-{cls}"):

            img_path = os.path.join(in_path, cls, img_name)
            img = cv2.imread(img_path)

            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gabor = apply_gabor(img, kernels)

            # NORMALIZACIÓN
            gabor = (gabor - gabor.mean()) / (gabor.std() + 1e-6)

            # Guardar como numpy
            save_path = os.path.join(out_path, cls, img_name.replace(".jpg", ".npy"))
            np.save(save_path, gabor)
````
DATASET PREPROCESADO
````
!python preprocesamiento.py
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

        for i, cls in enumerate(classes):
            for img in os.listdir(os.path.join(root_dir, cls)):
                self.paths.append(os.path.join(root_dir, cls, img))
                self.labels.append(i)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gabor = apply_gabor(img, self.kernels)

        # Normalización estable
        gabor = (gabor - gabor.mean()) / (gabor.std() + 1e-6)

        gabor = np.transpose(gabor, (2,0,1))

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
def gabor_kernel(sigma, theta, freq):
    # Kernel size EXACTO del paper
    N = int(2 * np.floor(3 * sigma) + 1)
    if N % 2 == 0:
        N += 1

    half = N // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]

    x_theta = x * np.cos(theta) + y * np.sin(theta)

    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    real = gaussian * np.cos(2 * np.pi * freq * x_theta)
    imag = gaussian * np.sin(2 * np.pi * freq * x_theta)

    return real.astype(np.float32), imag.astype(np.float32)


# =========================
# FILTER BANK (PAPER CONFIG)
# =========================
def build_gabor_kernels():
    kernels = []

    # Frecuencias del paper
    freqs = [0.1, 0.25, 0.4]

    # Orientaciones óptimas (MRI config del paper)
    orientations = [6, 8, 8]

    for f, n_theta in zip(freqs, orientations):

        # sigma derivado correctamente
        sigma = 1 / (2 * np.pi * f)

        thetas = np.linspace(0, np.pi, n_theta, endpoint=False)

        for theta in thetas:
            real, imag = gabor_kernel(sigma, theta, f)

            kernels.append((real, imag))

    return kernels


# =========================
# APPLY GABOR (PAPER PIPELINE)
# =========================
def apply_gabor(image, kernels):
    image = image.astype(np.float32)

    responses = []

    for real_k, imag_k in kernels:
        real = cv2.filter2D(image, cv2.CV_32F, real_k)
        imag = cv2.filter2D(image, cv2.CV_32F, imag_k)

        # MAGNITUDE (Eq. 6)
        mag = np.sqrt(real**2 + imag**2)
        responses.append(mag)

    # =========================
    # EXTRA CHANNELS (PAPER)
    # =========================

    # LPF
    lpf = cv2.GaussianBlur(image, (5,5), 0)
    responses.append(lpf)

    # HPF
    hpf = cv2.Laplacian(image, cv2.CV_32F)
    responses.append(hpf)

    responses = np.stack(responses, axis=-1)

    # =========================
    # ANTI-ALIASING
    # =========================
    responses = cv2.GaussianBlur(responses, (3,3), 0)

    # =========================
    # DOWNSAMPLING 224 → 56
    # =========================
    responses = responses[::4, ::4, :]

    return responses
````

MODELS 

gabor_cnn
````python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, c, r=2):
        super().__init__()
        self.fc1 = nn.Linear(c, c//r)
        self.fc2 = nn.Linear(c//r, c)

    def forward(self, x):
        b,c,_,_ = x.size()
        y = F.adaptive_avg_pool2d(x,1).view(b,c)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b,c,1,1)
        return x * y


class GaborCNN(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()

        self.se = SEBlock(in_channels)

        self.conv1 = nn.Conv2d(in_channels,128,3,padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(128,128,3,padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128,128,3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,num_classes)

    def forward(self,x):

        x = self.se(x)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        identity = x

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += identity
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.gap(x)
        x = torch.flatten(x,1)

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
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from models.gabor_cnn import GaborCNN
from utils.dataloader import GaborDataset

dataset = GaborDataset('/content/data/Training')

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample, _ = next(iter(train_loader))
model = GaborCNN(sample.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

best_loss = float('inf')
patience = 7
counter = 0

for epoch in range(100):

    # TRAIN
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1} | Train {train_loss/len(train_loader):.4f} | Val {val_loss:.4f}")

    # EARLY STOPPING
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        break
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

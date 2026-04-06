algorithm-based hyperparameter optimization

We try to reconstruct architecture to this article https://www.sciencedirect.com/science/article/pii/S209044792500632X using in the first part google colabs and then copy into local machine.

Project's structure
````
pso_cnn_project/
│
├── data/
├── models/
│   └── cnn.py
├── pso/
│   └── pso.py
├── train.py
├── evaluate.py
└── utils.py
````

```` bash
!mkdir -p /content/drive/MyDrive/pso_cnn_project

!mkdir -p /content/drive/MyDrive/pso_cnn_project/data
!mkdir -p /content/drive/MyDrive/pso_cnn_project/models
!mkdir -p /content/drive/MyDrive/pso_cnn_project/pso

!touch /content/drive/MyDrive/pso_cnn_project/models/cnn.py
!touch /content/drive/MyDrive/pso_cnn_project/pso/pso.py
!touch /content/drive/MyDrive/pso_cnn_project/train.py
!touch /content/drive/MyDrive/pso_cnn_project/evaluate.py
!touch /content/drive/MyDrive/pso_cnn_project/utils.py
````

## Select a upload Dataset
The dataset that we use to train the model is retrieved from https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image

````ipynb
%cd /content/drive/MyDrive/ga_cnn_project

!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

!unzip brain-tumor-mri-dataset.zip -d data/

!ls data/

!cp -r /content/drive/MyDrive/ga_cnn_project/data /content/data
````
The code in utils.py

````python
# utils.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_dataloaders(train_dir, test_dir, img_size=(224,224), batch_size=16):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=50,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_loader = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_loader = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_loader = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_loader, val_loader, test_loader
````
## Model CNN

`````python
# models/cnn.py

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224,224,3), num_classes=4, lr=1e-4, dropout=0.5, dense_units=128):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    x = Dense(dense_units, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # congelar backbone
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
``````
## Entrenamiento del modelo 

````python
# train.py

from models.cnn import build_model
from utils import get_dataloaders
from pso.pso import PSO
import tensorflow as tf

TRAIN_DIR = "/content/data/Training"
TEST_DIR = "/content/data/Testing"

def train():

    # PSO
    pso = PSO(num_particles=2, num_iterations=2)
    best_params, best_score = pso.optimize(TRAIN_DIR, TEST_DIR)

    print("Best params:", best_params)

    lr, dropout, dense, batch = best_params

    train_loader, val_loader, test_loader = get_dataloaders(
        TRAIN_DIR, TEST_DIR, batch_size=batch
    )

    model = build_model(
        lr=lr,
        dropout=dropout,
        dense_units=dense
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
    ]

    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=2, #30
        callbacks=callbacks
    )

    return model, test_loader


if __name__ == "__main__":
    train()
````
## Optimización de parametros con PSO

````python
# pso/pso.py

import numpy as np
from models.cnn import build_model
from utils import get_dataloaders

class PSO:

    def __init__(self, num_particles, num_iterations):
        self.num_particles = num_particles
        self.num_iterations = num_iterations

        # espacio de búsqueda
        self.lr_space = [1e-3, 1e-4, 1e-5]
        self.dropout_space = [0.3, 0.5, 0.7]
        self.dense_space = [64, 128, 256]
        self.batch_space = [8, 16, 32]

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            particle = [
              float(np.random.choice(self.lr_space)),
              float(np.random.choice(self.dropout_space)),
              int(np.random.choice(self.dense_space)),   
              int(np.random.choice(self.batch_space))    
            ]
            particles.append(particle)
        return particles

    def fitness(self, particle, train_dir, test_dir):

        lr, dropout, dense, batch = particle

        train_loader, val_loader, _ = get_dataloaders(
            train_dir, test_dir, batch_size=batch
        )

        model = build_model(
            lr=lr,
            dropout=dropout,
            dense_units=dense
        )

        history = model.fit(
          train_loader,
          validation_data=val_loader,
          epochs=2,
          steps_per_epoch=50,          
          validation_steps=20,         
          verbose=0
        )

        return max(history.history['val_accuracy'])

    def optimize(self, train_dir, test_dir):

        particles = self.initialize_particles()
        best_particle = None
        best_score = 0

        for i in range(self.num_iterations):
            print(f"Iteración {i+1}")

            for particle in particles:
                score = self.fitness(particle, train_dir, test_dir)

                if score > best_score:
                    best_score = score
                    best_particle = particle

            print("Best score:", best_score)

        return best_particle, best_score
````
## Evaluate model
````python
# evaluate.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_loader):

    preds = model.predict(test_loader)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_loader.classes

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
````
## Flujo de ejecución 
`````python
import sys
sys.path.append('/content/drive/MyDrive/pso_cnn_project')
``````
````python
!cd /content/drive/MyDrive/pso_cnn_project
````
````python
!python /content/drive/MyDrive/pso_cnn_project/train.py
````
````python
from evaluate import evaluate_model
from tensorflow.keras.models import load_model
from utils import get_dataloaders  

model = load_model("best_model.h5")

_, _, test_loader = get_dataloaders("data/Training", "data/Testing")

evaluate_model(model, test_loader)
````

# Modificado para CXR

## Project's structure
````
pso_cnn_project/
│
├── data/
├── models/
│   └── cnn.py
├── pso/
│   └── pso.py
├── train.py
├── evaluate.py
└── utils.py
````

```` bash
!mkdir -p /content/drive/MyDrive/pso_cnn_project

!mkdir -p /content/drive/MyDrive/pso_cnn_project/data
!mkdir -p /content/drive/MyDrive/pso_cnn_project/models
!mkdir -p /content/drive/MyDrive/pso_cnn_project/pso

!touch /content/drive/MyDrive/pso_cnn_project/models/cnn.py
!touch /content/drive/MyDrive/pso_cnn_project/pso/pso.py
!touch /content/drive/MyDrive/pso_cnn_project/train.py
!touch /content/drive/MyDrive/pso_cnn_project/evaluate.py
!touch /content/drive/MyDrive/pso_cnn_project/utils.py
````

## Select a upload Dataset
The dataset that we use to train the model is retrieved from https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image

````ipynb
%cd /content/drive/MyDrive/ga_cnn_project

!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d sachinkumar413/covid-pneumonia-normal-chest-xray-images

!unzip covid-pneumonia-normal-chest-xray-images.zip -d data/

!ls data/

!cp -r /content/drive/MyDrive/ga_cnn_project/data /content/data
````
The code in until.py

````python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_dataloaders(data_dir, batch_size=32, img_size=224):

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Dataset completo (sin Training/Test)
    full_dataset = datasets.ImageFolder(
        root=f"{data_dir}",
        transform=train_transform
    )

    # Split: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    # cambiar transform de test/val
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def get_scheduler(optimizer, patience=5, factor=0.5, min_lr=1e-7):
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )
````
## Model CNN

`````python
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
``````
## Entrenamiento del modelo 

````python
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import CNNModel
from utils import get_dataloaders, get_scheduler

def train_model(data_dir, lr=0.001, batch_size=32, epochs=2,
                factor=0.5, min_lr=1e-7, patience=5): #50 epo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.85, 0.9925))
    scheduler = get_scheduler(optimizer, patience=patience, factor=factor, min_lr=min_lr)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # VALIDACIÓN
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss/len(train_loader):.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    return model, best_val_loss
````
## Optimización de parametros con WOA

````python
import numpy as np
from train import train_model

class WhaleOptimizer:
    def __init__(self, data_dir, n_whales=5, max_iter=10):
        self.data_dir = data_dir
        self.n_whales = n_whales
        self.max_iter = max_iter

        self.dim = 4
        self.bounds = [(1e-5, 1e-2), (0.1, 0.5), (1e-7, 1e-4), (16, 64)]

        self.positions = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(n_whales, self.dim)
        )

        self.fitness = np.full(n_whales, np.inf)
        self.best_pos = None
        self.best_fitness = np.inf
        self.cache = {}

    def objective(self, pos):
        key = tuple(np.round(pos, 6))  

        if key in self.cache:
            return self.cache[key]

        lr, factor, min_lr, batch_size = pos
        batch_size = int(round(batch_size))

        try:
            _, val_loss = train_model(
                self.data_dir,
                lr=lr,
                batch_size=batch_size,
                epochs=2, #10
                factor=factor,
                min_lr=min_lr,
                patience=3
            )

            self.cache[key] = val_loss
            return val_loss

        except Exception as e:
            print(f"Error con {pos}: {e}")
            return 1e6

    def optimize(self):
        # Inicialización
        for i in range(self.n_whales):
            self.fitness[i] = self.objective(self.positions[i])

            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_pos = self.positions[i].copy()

        for t in range(self.max_iter):
            a = 2 - 2 * t / self.max_iter

            for i in range(self.n_whales):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.random()

                if p < 0.5:
                    if abs(A[0]) < 1:
                        D = abs(C * self.best_pos - self.positions[i])
                        new_pos = self.best_pos - A * D
                    else:
                        rand_idx = np.random.randint(0, self.n_whales)
                        D = abs(C * self.positions[rand_idx] - self.positions[i])
                        new_pos = self.positions[rand_idx] - A * D
                else:
                    b = 1
                    l = np.random.uniform(-1, 1)
                    D = abs(self.best_pos - self.positions[i])
                    new_pos = D * np.exp(b * l) * np.cos(2*np.pi*l) + self.best_pos

                new_pos = np.clip(
                    new_pos,
                    [b[0] for b in self.bounds],
                    [b[1] for b in self.bounds]
                )

                new_pos[3] = int(round(new_pos[3]))

                new_fit = self.objective(new_pos)

                if new_fit < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fit

                    if new_fit < self.best_fitness:
                        self.best_fitness = new_fit
                        self.best_pos = new_pos.copy()

            print(f"WOA iter {t+1}/{self.max_iter} | best loss: {self.best_fitness:.6f}")

        return self.best_pos, self.best_fitness
````
## Evaluate model
````python
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, class_names=None):
    import torch
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if class_names is None:
        class_names = test_loader.dataset.dataset.classes 

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    print("\n=== Resultados por clase ===")
    for i, name in enumerate(class_names):
        print(f"{name:12} | Prec: {precision[i]:.4f} | Rec: {recall[i]:.4f} | F1: {f1[i]:.4f}")

    macro_prec = np.mean(precision)
    macro_rec = np.mean(recall)
    macro_f1 = np.mean(f1)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_prec:.4f}")
    print(f"Macro Recall:    {macro_rec:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')

    plt.title("Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

    return {
        "accuracy": acc,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1
    }
````
## Flujo de ejecución 
`````python
import sys
sys.path.append('/content/drive/MyDrive/woa_cnn_project')
``````
````python
!cd /content/drive/MyDrive/woa_cnn_project
````
````python
from woa.whale_optimization_algorithm import WhaleOptimizer
woa = WhaleOptimizer(data_dir="/content/data", n_whales=2, max_iter=2)
best_hp, best_loss = woa.optimize()
print("\n=== Mejores hiperparámetros encontrados ===")
print(f"lr = {best_hp[0]:.6f}")
print(f"factor = {best_hp[1]:.3f}")
print(f"min_lr = {best_hp[2]:.2e}")
print(f"batch_size = {int(best_hp[3])}")
````
````python
from train import train_model
model, _ = train_model("/content/data",
                       lr=best_hp[0],
                       batch_size=int(best_hp[3]),
                       epochs=10,
                       factor=best_hp[1],
                       min_lr=best_hp[2],
                       patience=5)
````
````python
from evaluate import evaluate_model
from utils import get_dataloaders
_, _, test_loader = get_dataloaders("/content/data", batch_size=int(best_hp[3]))
results = evaluate_model(model, test_loader)
````

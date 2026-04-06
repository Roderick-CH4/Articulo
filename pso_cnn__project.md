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
!cd /content

!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d sachinkumar413/covid-pneumonia-normal-chest-xray-images

!unzip covid-pneumonia-normal-chest-xray-images.zip -d data/

!ls data/

````
The code in until.py

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

def build_model(input_shape=(224,224,3), num_classes=3, lr=1e-4, dropout=0.5, dense_units=128):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    x = Dense(int(dense_units), activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # congelar casi todo
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
``````
## Entrenamiento del modelo 

````python
# train.py

import tensorflow as tf
from models.cnn import build_model
from utils import get_dataloaders
from pso.pso import PSO

TRAIN_DIR = "/content/data_split/train"
TEST_DIR  = "/content/data_split/test"

def train():

    print("Iniciando PSO")

    pso = PSO(num_particles=3, num_iterations=2)
    best_params, best_score = pso.optimize(TRAIN_DIR, TEST_DIR)

    print("\nBest params:", best_params)

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

    print("\nEntrenamiento final")

    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=10,
        callbacks=callbacks,
        verbose=1
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

        self.lr_space = [1e-3, 1e-4, 1e-5]
        self.dropout_space = [0.3, 0.5, 0.7]
        self.dense_space = [64, 128, 256]
        self.batch_space = [8, 16]

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

        score = max(history.history['val_accuracy'])
        print(f"Params: {particle} → Acc: {score}")

        return score

    def optimize(self, train_dir, test_dir):

        particles = self.initialize_particles()
        best_particle = None
        best_score = 0

        for i in range(self.num_iterations):
            print(f"\nIteración {i+1}")

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

    print("\nClasses:", test_loader.class_indices)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
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
import os
import shutil
import random

base_dir = "/content/data"
output_dir = "/content/data_split"

classes = ["COVID", "NORMAL", "PNEUMONIA"]

for cls in classes:
    os.makedirs(f"{output_dir}/train/{cls}", exist_ok=True)
    os.makedirs(f"{output_dir}/test/{cls}", exist_ok=True)

    images = os.listdir(f"{base_dir}/{cls}")
    random.shuffle(images)

    split = int(0.8 * len(images))

    train_imgs = images[:split]
    test_imgs = images[split:]

    for img in train_imgs:
        shutil.copy(f"{base_dir}/{cls}/{img}", f"{output_dir}/train/{cls}/{img}")

    for img in test_imgs:
        shutil.copy(f"{base_dir}/{cls}/{img}", f"{output_dir}/test/{cls}/{img}")

print("Dataset separado correctamente")
````
````python
!python /content/drive/MyDrive/pso_cnn_project/train.py
````
````python
from evaluate import evaluate_model
from tensorflow.keras.models import load_model
from utils import get_dataloaders

model = load_model("best_model.h5")

_, _, test_loader = get_dataloaders(
    "/content/data_split/train",
    "/content/data_split/test"
)

evaluate_model(model, test_loader)
````

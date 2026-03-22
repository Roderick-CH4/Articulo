# Reconstruct General CNN model for biomedical image classification via genetic 
algorithm-based hyperparameter optimization

We try to reconstruct architecture to this article https://www.sciencedirect.com/science/article/pii/S209044792500632X
using in the first part google colabs and then copy into local machine.

### Project's structure
```` 
ga_cnn_project/
│
├── data/
├── models/
│   └── cnn.py
├── ga/
│   └── genetic_algorithm.py
├── train.py
├── evaluate.py
└── utils.py
````

````ipynb
!mkdir -p /content/drive/MyDrive/ga_cnn_project

!mkdir -p /content/drive/MyDrive/ga_cnn_project/data
!mkdir -p /content/drive/MyDrive/ga_cnn_project/models
!mkdir -p /content/drive/MyDrive/ga_cnn_project/ga

!touch /content/drive/MyDrive/ga_cnn_project/models/cnn.py
!touch /content/drive/MyDrive/ga_cnn_project/ga/genetic_algorithm.py
!touch /content/drive/MyDrive/ga_cnn_project/train.py
!touch /content/drive/MyDrive/ga_cnn_project/evaluate.py
!touch /content/drive/MyDrive/ga_cnn_project/utils.py
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

%cd data

!unzip covid-pneumonia-normal-chest-xray-images.zip -d data/

!ls data/
````
The code in until.py
````python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32):

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split: 64% train, 16% val, 20% test
    total_size = len(dataset)
    train_size = int(0.64 * total_size)
    val_size = int(0.16 * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader
````

Checke split

````python
from utils import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders('/content/drive/MyDrive/ga_cnn_project/data')

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break
````

## Model CNN

Esto es lo mas apegado al documento que pude conseguir, revisar con Aldo

````python
import torch
import torch.nn as nn



class CNNModel(nn.Module):

    def _initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def __init__(self, params, num_classes=3):
        super(CNNModel, self).__init__()

        filters = params["filters"]
        kernel_size = params["kernel"]
        dropout = params["dropout"]

        # Activación
        if params["activation"] == "relu":
            self.activation = nn.ReLU()
        elif params["activation"] == "tanh":
            self.activation = nn.Tanh()
        elif params["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif params["activation"] == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        padding = kernel_size // 2 if params["padding"] == "same" else 0

        # Bloques CNN
        self.conv1 = nn.Conv2d(3, filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(filters * 2)

        self.conv3 = nn.Conv2d(filters * 2, filters * 4, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(filters * 4)

        self.conv4 = nn.Conv2d(filters * 4, filters * 8, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(filters * 8)

        self.pool = nn.MaxPool2d(params["pool_size"])

        # CLAVE: SIEMPRE reduce a 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # Fully Connected
        self.fc1 = nn.Linear(filters * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def forward(self, x):

        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn4(self.conv4(x))))

        # SIEMPRE antes del flatten
        x = self.global_pool(x)

        # flatten seguro
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
````

Check model 

````python
import torch 
from models.cnn import CNNModel

params = {
    "filters": 32,
    "kernel": 3,
    "dropout": 0.25,
    "activation": "relu",
    "padding": "same",
    "pool_size": 2
}

model = CNNModel(params)

x = torch.randn(1, 3, 150, 150)
y = model(x)

print(y.shape)
````

Entrenamiento del modelo 

````python
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNNModel
from utils import get_dataloaders


def train_model(params, data_dir, epochs=5):

    device = torch.device("cuda")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=params["batch_size"]
    )

    # Modelo
    model = CNNModel(params).to(device)

    # Loss y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    # Validación (fitness para GA después)
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss:.4f}")

    return model, val_loss


if __name__ == "__main__":

    params = {
        "filters": 32,
        "kernel": 3,
        "dropout": 0.25,
        "activation": "relu",
        "padding": "same",
        "pool_size": 2,
        "batch_size": 16
    }

    model, val_loss = train_model(
        params,
        data_dir="/content/drive/MyDrive/ga_cnn_project/data",
        epochs=5
    )
````

## El super mutante algoritmo de genes 
Algoritmo genetico lo mas apegado al documento

`````python
import random
from train import train_model

param_space = {
    "filters": [8, 16, 32, 64, 128, 256, 512],
    "kernel": [2, 3, 4, 5, 6, 7],
    "dropout": [0, 0.25, 0.5, 0.75],
    "activation": ["relu", "tanh", "elu", "leaky_relu"],
    "padding": ["same", "valid"],
    "pool_size": [2, 3, 4],
    "batch_size": [4, 8, 16, 32]
}

def create_individual():
    return {key: random.choice(values) for key, values in param_space.items()}

def create_population(size):
    return [create_individual() for _ in range(size)]

def is_valid(individual):
    if individual["kernel"] > 5 and individual["pool_size"] > 2:
        return False
    return True

def fitness(individual, data_dir):
    if not is_valid(individual):
        return float("inf")
    try:
        _, val_loss = train_model(individual, data_dir, epochs=5)
        return val_loss
    except:
        return float("inf")

def selection(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), k)
    selected = sorted(selected, key=lambda x: x[1])
    return selected[0][0]

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual, mutation_rate=0.1):
    for key in individual:
        if random.random() < mutation_rate:
            individual[key] = random.choice(param_space[key])
    return individual

def genetic_algorithm(data_dir, population_size=10, generations=5):

    population = create_population(population_size)

    best_individual = None
    best_score = float("inf")

    for gen in range(generations):
        print(f"\nGeneración {gen+1}/{generations}")

        scores = []

        for individual in population:
            print(f"Evaluando: {individual}")
            score = fitness(individual, data_dir)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_individual = individual

        new_population = []

        for _ in range(population_size):
            parent1 = selection(population, scores)
            parent2 = selection(population, scores)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    print("\nMejor configuración encontrada:")
    print(best_individual)
    print("Validation Loss:", best_score)

    return best_individual
`````

## training and optimized
entrenamiento 

````python
!python train.py
````

optimizacion con GA

````pyhton
from ga.genetic_algorithm import genetic_algorithm

best = genetic_algorithm(
    data_dir="/content/drive/MyDrive/ga_cnn_project/data",  # o tu ruta
    population_size=5,
    generations=3
)
````

## Evaluate 
Evaluamos el mejor resultado del GA

````python
best = genetic_algorithm(
    data_dir="/content/drive/MyDrive/ga_cnn_project/data"
)

print("Mejor configuración:", best)
````

````python
from train import train_model
from evaluate import evaluate_model
from utils import get_dataloaders

# mejores parámetros del GA
best_params = best  # o los copias manualmente

# cargar datos
train_loader, val_loader, test_loader = get_dataloaders('/content/drive/MyDrive/ga_cnn_project/data')

# entrenamiento final (más epochs)
model, _ = train_model(best_params, '/content/drive/MyDrive/ga_cnn_project/data', epochs=20)

# evaluación
evaluate_model(model, test_loader)
````


-------------------------------------------------------------------------------------------------------------------------

-----------------------


-------------
## Genetic Algorithm 

Problemas que solucionar 

````python
import random
from train import train_model


# Espacio de búsqueda (del paper)

param_space = {
    "filters": [8, 16, 32, 64, 128, 256],
    "kernel": [3, 4, 5],
    "dropout": [0, 0.25, 0.5],
    "activation": ["relu", "tanh", "elu", "leaky_relu"],
    "padding": ["same", "valid"],
    "pool_size": [2, 3],
    "batch_size": [8, 16, 32]
}

"""
# Espacio de busqueda ajustado para evitar downsampling
param_space = {
    "filters": [8, 16, 32, 64, 128, 256],
    "kernel": [3],          
    "dropout": [0, 0.25, 0.5],
    "activation": ["relu", "tanh", "elu", "leaky_relu"],
    "padding": ["same"],      
    "pool_size": [2],          
    "batch_size": [8, 16, 32]
}
"""
# Crear individuo aleatorio
def create_individual():
    return {key: random.choice(values) for key, values in param_space.items()}


# Crear población
def create_population(size):
    return [create_individual() for _ in range(size)]


# Evaluar individuo

def fitness(individual, data_dir):
    _, val_loss = train_model(individual, data_dir, epochs=3)  # epochs bajos
    return val_loss

# Evitar downsampling
"""
def fitness(individual, data_dir):
    try:
        _, val_loss = train_model(individual, data_dir, epochs=3)
        return val_loss
    except:
        return float("inf")  # penaliza configuraciones malas

"""
# Selección (torneo)
def selection(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), k)
    selected = sorted(selected, key=lambda x: x[1])
    return selected[0][0]


# Crossover
def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


# Mutación
def mutate(individual, mutation_rate=0.1):
    for key in individual:
        if random.random() < mutation_rate:
            individual[key] = random.choice(param_space[key])
    return individual


# ALGORITMO PRINCIPAL
def genetic_algorithm(data_dir, population_size=10, generations=5):

    population = create_population(population_size)

    for gen in range(generations):
        print(f"\n Generación {gen+1}/{generations}")

        scores = []

        for individual in population:
            print(f"Evaluando: {individual}")
            score = fitness(individual, data_dir)
            scores.append(score)

        # Selección de mejores
        new_population = []

        for _ in range(population_size):
            parent1 = selection(population, scores)
            parent2 = selection(population, scores)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    # Mejor resultado final
    final_scores = [fitness(ind, data_dir) for ind in population]
    best_idx = final_scores.index(min(final_scores))

    best_individual = population[best_idx]

    print("\n Mejor configuración encontrada:")
    print(best_individual)

    return best_individual
````
## Modelo CNN

Problemas que solucionar

````python
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, params, num_classes=3):
        super(CNNModel, self).__init__()

        filters = params["filters"]
        kernel_size = params["kernel"]
        dropout = params["dropout"]

        # Activación
        if params["activation"] == "relu":
            self.activation = nn.ReLU()
        elif params["activation"] == "tanh":
            self.activation = nn.Tanh()
        elif params["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif params["activation"] == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        padding = kernel_size // 2 if params["padding"] == "same" else 0

        # Bloques CNN
        self.conv1 = nn.Conv2d(3, filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(filters * 2)

        self.conv3 = nn.Conv2d(filters * 2, filters * 4, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(filters * 4)

        self.conv4 = nn.Conv2d(filters * 4, filters * 8, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(filters * 8)

        self.pool = nn.MaxPool2d(params["pool_size"])
        self.dropout = nn.Dropout(dropout)

        # CLAVE: SIEMPRE reduce a 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected
        self.fc1 = nn.Linear(filters * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn4(self.conv4(x))))
        x = self.dropout(x)

        # SIEMPRE antes del flatten
        x = self.global_pool(x)

        # flatten seguro
        x = torch.flatten(x, 1)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
````
## Entrenamiento 

````python
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNNModel
from utils import get_dataloaders


def train_model(params, data_dir, epochs=5):

    device = torch.device("cuda")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=params["batch_size"]
    )

    # Modelo
    model = CNNModel(params).to(device)

    # Loss y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    # Validación (fitness para GA después)
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss:.4f}")

    return model, val_loss


if __name__ == "__main__":

    params = {
        "filters": 32,
        "kernel": 3,
        "dropout": 0.25,
        "activation": "relu",
        "padding": "same",
        "pool_size": 2,
        "batch_size": 16
    }

    model, val_loss = train_model(
        params,
        data_dir="/content/drive/MyDrive/ga_cnn_project/data",
        epochs=5
    )
````
## CNN y GA mas apegados al documento 

Por algun motivo me dio peores resultados de loss en test pequeño 

### CNN
````python
import torch
import torch.nn as nn



class CNNModel(nn.Module):

    def _initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def __init__(self, params, num_classes=3):
        super(CNNModel, self).__init__()

        filters = params["filters"]
        kernel_size = params["kernel"]
        dropout = params["dropout"]

        # Activación
        if params["activation"] == "relu":
            self.activation = nn.ReLU()
        elif params["activation"] == "tanh":
            self.activation = nn.Tanh()
        elif params["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif params["activation"] == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        padding = kernel_size // 2 if params["padding"] == "same" else 0

        # Bloques CNN
        self.conv1 = nn.Conv2d(3, filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(filters * 2)

        self.conv3 = nn.Conv2d(filters * 2, filters * 4, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(filters * 4)

        self.conv4 = nn.Conv2d(filters * 4, filters * 8, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(filters * 8)

        self.pool = nn.MaxPool2d(params["pool_size"])

        # CLAVE: SIEMPRE reduce a 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # Fully Connected
        self.fc1 = nn.Linear(filters * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def forward(self, x):

        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.pool(self.activation(self.bn4(self.conv4(x))))

        # SIEMPRE antes del flatten
        x = self.global_pool(x)

        # flatten seguro
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
````

### GA
````python
import random
from train import train_model

param_space = {
    "filters": [8, 16, 32, 64, 128, 256, 512],
    "kernel": [2, 3, 4, 5, 6, 7],
    "dropout": [0, 0.25, 0.5, 0.75],
    "activation": ["relu", "tanh", "elu", "leaky_relu"],
    "padding": ["same", "valid"],
    "pool_size": [2, 3, 4],
    "batch_size": [4, 8, 16, 32]
}

def create_individual():
    return {key: random.choice(values) for key, values in param_space.items()}

def create_population(size):
    return [create_individual() for _ in range(size)]

def is_valid(individual):
    if individual["kernel"] > 5 and individual["pool_size"] > 2:
        return False
    return True

def fitness(individual, data_dir):
    if not is_valid(individual):
        return float("inf")
    try:
        _, val_loss = train_model(individual, data_dir, epochs=5)
        return val_loss
    except:
        return float("inf")

def selection(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), k)
    selected = sorted(selected, key=lambda x: x[1])
    return selected[0][0]

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual, mutation_rate=0.1):
    for key in individual:
        if random.random() < mutation_rate:
            individual[key] = random.choice(param_space[key])
    return individual

def genetic_algorithm(data_dir, population_size=10, generations=5):

    population = create_population(population_size)

    best_individual = None
    best_score = float("inf")

    for gen in range(generations):
        print(f"\nGeneración {gen+1}/{generations}")

        scores = []

        for individual in population:
            print(f"Evaluando: {individual}")
            score = fitness(individual, data_dir)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_individual = individual

        new_population = []

        for _ in range(population_size):
            parent1 = selection(population, scores)
            parent2 = selection(population, scores)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    print("\nMejor configuración encontrada:")
    print(best_individual)
    print("Validation Loss:", best_score)

    return best_individual
````

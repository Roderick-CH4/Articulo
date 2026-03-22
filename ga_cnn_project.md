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

### Select a upload Dataset
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




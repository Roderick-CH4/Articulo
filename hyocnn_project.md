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
!touch /content/drive/MyDrive/pso_cnn_project/optimization/pso.py
!touch /content/drive/MyDrive/pso_cnn_project/train.py
!touch /content/drive/MyDrive/pso_cnn_project/evaluate.py
!touch /content/drive/MyDrive/pso_cnn_project/utils.py
````

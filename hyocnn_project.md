algorithm-based hyperparameter optimization

We try to reconstruct architecture to this article https://www.sciencedirect.com/science/article/pii/S209044792500632X using in the first part google colabs and then copy into local machine.

Project's structure
````
hyocnn_project/
│
├── data/
├── models/
│   └── xception_model.py
├── optimization/
│   ├── pso.py
│   ├── ga.py
│   └── hybrid_pso_ga.py
├── train.py
├── evaluate.py
└── utils.py
````

```` bash
!mkdir -p /content/drive/MyDrive/hyocnn_project

!mkdir -p /content/drive/MyDrive/hyocnn_project/data
!mkdir -p /content/drive/MyDrive/hyocnn_project/models
!mkdir -p /content/drive/MyDrive/hyocnn_project/optimization

!touch /content/drive/MyDrive/hyocnn_project/models/xception_model.py
!touch /content/drive/MyDrive/hyocnn_project/optimization/pso.py
!touch /content/drive/MyDrive/hyocnn_project/optimization/ga.py
!touch /content/drive/MyDrive/hyocnn_project/optimization/hybrid_pso_ga.py
!touch /content/drive/MyDrive/hyocnn_project/train.py
!touch /content/drive/MyDrive/hyocnn_project/evaluate.py
!touch /content/drive/MyDrive/hyocnn_project/utils.py
````

---
# 1

## Un enfoque híbrido ligero de aprendizaje profundo de Gabor y su aplicación a la clasificación de imágenes médicas
## A Lightweight Hybrid Gabor Deep Learning Approach and its Application to Medical Image Classification

### Fecha 2026

https://link.springer.com/article/10.1007/s11263-025-02658-2#Sec2

### Algoritmos
Modelo híbrido Gabor-CNN
### Datasets
*Todos estos Datasets son públicos*

| Conjunto de datos                          | Modalidad                               | #Classes | #Samples |
|---------------------------------------------|------------------------------------------|---------|---------|
| Resonancia magnética de tumor cerebral     | Resonancia magnética (IRM)               | 4       | 7,023   |
| DermaMNIST                                  | Dermatoscopio                            | 7       | 10,015  |
| BloodMNIST                                  | Microscopio de células sanguíneas        | 8       | 17,092  |
| OrganCMNIST                                 | TAC abdominal                            | 11      | 23,583  |
| COVID-QU-Ex                                 | Radiografía de tórax (CXR)               | 3       | 33,026  |

---
# 2

## Análisis comparativo de arquitecturas híbridas CNN-Transformer para la clasificación de imágenes médicas en múltiples tipos de cáncer.
## Comparative Analysis of Hybrid CNN-Transformer Architectures for Medical Image Classification Across Multiple Cancer Types

### Fecha 2026

https://www.researchgate.net/profile/Afex-Callagher/publication/399466221_Comparative_Analysis_of_Hybrid_CNN-Transformer_Architectures_for_Medical_Image_Classification_Across_Multiple_Cancer_Types/links/695c28af27359023a013c9c7/Comparative-Analysis-of-Hybrid-CNN-Transformer-Architectures-for-Medical-Image-Classification-Across-Multiple-Cancer-Types.pdf

### Algoritmos
- Pure CNNs: ResNet-50 and DenseNet-121, pretrained on ImageNet. 
- Pure Transformer: Vision Transformer Base/16 (ViT-B/16), pretrained on 
ImageNet-21k. 
- Hybrid Models: CvT-13, MobileViT-XS, and a custom LiteHybridNet (a 
streamlined model with 4 convolutional blocks followed by a 4-layer 
Transformer encoder).

### Datasets
| Dataset | Tipo de cáncer / órgano | Modalidad de imagen | #Clases | #Muestras | Descripción |
|--------|--------------------------|---------------------|--------|-----------|-------------|
| LC25000 (Lung & Colon) | Pulmón y colon | Histopatología | 5 | 25,000 | Imágenes de tejido pulmonar y de colon: lung benign, lung adenocarcinoma, lung squamous cell carcinoma, colon benign, colon adenocarcinoma (Borkowski et al., 2019). |
| BreakHis (Breast) | Mama | Microscopía histopatológica | 8 | 9,109 | Imágenes microscópicas de tumores de mama tomadas a cuatro niveles de aumento (4 benignas y 4 malignas) (Spanhol et al., 2016). |
| HAM10000 (Skin) | Piel | Dermatoscopía | 7 | 10,015 | Colección de imágenes dermatoscópicas de lesiones pigmentadas de la piel (Tschandl et al., 2018). |
| PCam (Metastasis Detection) | Metástasis en ganglios linfáticos | Histopatología (patches) | 2 | 327,680 | Dataset basado en parches para detección de cáncer metastásico de mama en secciones de ganglios linfáticos (Veeling et al., 2018). |
| Brain Tumor MRI (Brain) | Tumor cerebral | Resonancia Magnética (MRI) | 4 | 7,023 | Imágenes MRI T1 con contraste con clases: glioma, meningioma, pituitary tumor y no tumor (Cheng et al., 2016). |

---
# 3

## Una comparación experimental de modelos de aprendizaje profundo para la clasificación de neumonías a partir de imágenes de radiografías de tórax
## An experimental comparison of deep learning models for pneumonia classification from chest X-ray images

https://www.sciencedirect.com/science/article/pii/S1746809425012534

### Fecha 2026, Pero ocupa modelos de entre 2019-2021

### Algoritmos 
| Modelo       | Precisión (%) | Precisión Neumonía (%) | Precisión COVID-19 (%) | Precisión Normal (%) | Revocación Neumonía (%) | Revocación COVID-19 (%) | Revocación Normal (%) | F1 Neumonía (%) | F1 COVID-19 (%) | F1 Normal (%) |
|--------------|---------------|------------------------|------------------------|----------------------|--------------------------|--------------------------|-----------------------|-----------------|-----------------|---------------|
| ResNet       | 95            | 92                     | 90                     | 96                   | 91                       | 88                       | 95                    | 91              | 89              | 97            |
| AlexNet      | 90            | 88                     | 85                     | 92                   | 87                       | 84                       | 91                    | 87              | 84              | 91            |
| VGG-19       | 97            | 96                     | 91                     | 99                   | 98                       | 97                       | 88                    | 97              | 93              | 94            |
| SqueezeNet   | 88            | 85                     | 82                     | 90                   | 83                       | 80                       | 89                    | 84              | 81              | 89            |
| DenseNet     | 92            | 89                     | 87                     | 95                   | 88                       | 85                       | 94                    | 88              | 86              | 95            |
| InceptionV3  | 94            | 91                     | 89                     | 95                   | 90                       | 86                       | 96                    | 90              | 87              | 95            |

### Datasets
Utilizamos el conjunto de datos disponible públicamente del conjunto de datos Kaggle, compuesto por 6939 imágenes de rayos X

--- 
# 4

## Aprendizaje profundo explicable para la detección de neumonía pediátrica en imágenes de radiografía de tórax
## Explainable Deep Learning for Pediatric Pneumonia Detection in Chest X-Ray Images

### Fecha 2026

https://arxiv.org/pdf/2601.09814

### Algoritmos
Dense Net121 and EfficientNet-B0

### Datasets
chest X-ray dataset introduced by Kermany, The dataset consists of 5,863 anterior–posterior (AP) chest radio
graphs from children between the ages of one and five years

---
# 5 

## Un Marco Híbrido de Aprendizaje Profundo para el Diagnóstico Automatizado de Trastornos Dentales a partir de Imágenes de Rayos X
## A Hybrid Deep Learning Framework for Automated Dental Disorder Diagnosis from X-Ray Images

### Fecha 2026

https://digitalmanuscriptpedia.com/conferences/index.php/DMP-LNMR/article/download/95/95v

### Algoritmos
It combined HOG as handcrafted descriptors with DenseNet-201 and the Swin Transformer for transformer-based features, capturing complementary information and encompassing fine-grained low-level spatial characteristics as well as rich high-level semantic representations. 

### Datasets
The DRAD dataset focused on dental radiography analysis and diagnosis. The dataset consists of 1272 X-ray images. (No estoy seguro que sea publico)

---
# 6

## Modelo CNN general para la clasificación de imágenes biomédicas mediante optimización de hiperparámetros basada en algoritmos genéticos
## General CNN model for biomedical image classification via genetic algorithm-based hyperparameter optimization

### Fecha 2026

https://www.sciencedirect.com/science/article/pii/S209044792500632X

### Algoritmos
CNN optimized via Genetic Algorithm (GA)

### Datasets
we use three biomedical datasets: MS, Alzheimer’s, and COVID-19 datasets. While the MS dataset is used for GA to optimize the CNN hyperparameters and test the optimized CNN, Alzheimer’s and COVID-19 are used only for the generalization test of the optimized CNN.

---
# 7

## HyOCNN: Red neuronal convolucional optimizada híbrida para clasificación de imágenes robusta
## HyOCNN: Hybrid-optimized convolutional neural network for robust imageclassification

### Fecha 2026

https://www.sciencedirect.com/science/article/pii/S1746809425015514

### Algoritmos
The proposed model utilizes the Xception architecture, To optimize modelperformance, a hybrid PSO-GA. Finally this hybrid modedl was named the optimized HyOCNN architecture.

### Datasets
The original dataset wascreated and collected by Khan et al. [28]1, and named Dataset-1 (asdescribed in Table 1), this dataset was used to train, validate and testour proposed HyOCNN model.

The rest of the three datasets were onlytested to further test the generalization and the resilience of HyOCNN.They gathered a sample of publicly available databases with chest X-ray images on GitHub and the Kaggle repositories of four categories:COVID-19, normal, bacterial pneumonia, and viral pneumonia.

| Categoría | Dataset-1 Train | Dataset-1 Test | Dataset-1 Total | Dataset-2 | Dataset-3 | Dataset-4 | Test |
|-----------|----------------|---------------|---------------|-----------|-----------|-----------|------|
| COVID-19 | 224 | 60 | 284 | 341 | 341 | 341 | 45 |
| Normal | 237 | 73 | 310 | 2800 | – | – | 76 |
| Bacterial pneumonia | 256 | 74 | 330 | – | – | 2772 | 74 |
| Viral pneumonia | 257 | 70 | 327 | – | 1493 | – | 63 |
| **Total** | **974** | **277** | **1251** | **3141** | **1834** | **3113** | **258** |

---
# 8

## Clasificación radiográfica de tumores óseos primarios impulsada por aprendizaje profundo utilizando modelos híbridos aumentados con atención
## Deep learning driven radiographic classification of primary bone tumorsusing attention augmented hybrid models

### Fecha 2026

https://www.sciencedirect.com/science/article/pii/S1746809425013990

### Algoritmos
Convolutional Neural Networks Transformer (CNNT) model --> CNN + Transformer Model
ResNet50 and Convolutional Block Attention Module (CBAM)model --> ResNet50 + CBAM Model Training for Bone TumorClassification

### Datasets
In this work, we used the BTXRD dataset, an open dataset for theclassification, localization, and segmentation of primary bone tumorsfrom radiographs [13]. The dataset consists of 3746 X-ray images,including 1879 standard bone images, 1525 benign bone tumor images,and 342 malignant bone tumor images.

---
# 9

## Bone-CNN: Una Arquitectura de Aprendizaje Profundo Ligera para la Clasificación Multiclase de Tumores Óseos Primarios en Radiografías
## Bone-CNN: A Lightweight Deep Learning Architecture for Multi-Class Classification of Primary Bone Tumours in Radiographs

### Fecha 2026

https://www.mdpi.com/2227-9059/14/2/299

### Algoritmos
Thep roposed Bone-CNN refers to a lightweight CNN architecture optimised for bone tumour image classification, using depthwise separable convolutions, multi-scale
learning, and compact classification heads to focus on the efficient extraction of features for primary bone tumour classification on radiographic images.

### Datasets
This study uses the publicly available Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors, introduced by Yao et al. [22]. The original collection contains 884 anonymised radiographs across nine primary bone tumour categories, sourced from multiple imaging centres and released under a CC-BY 4.0 licence.

---
# 10

## Arquitecturas de aprendizaje profundo en conjunto para detectar tuberculosis pulmonar en radiografías de tórax
## Ensemble deep learning architectures for detecting pulmonary tuberculosis in chest X-rays

### Fecha 2026

https://www.nature.com/articles/s41598-025-30792-x

### Algoritmos
Convolutional Autoencoder (CAE-NN)
This architecture enables the CAE-NN to learn compact, high-level representations of the CXRs, contributing to  the overall robustness and performance of the TB diagnostic process.

Multi-Scale CNN (MS-CNN)
A multi-scale ResNet34 with deep layer aggregation35 is proposed for improved feature extraction and classification. 

Actually use a hybrid modedl combined Convolutional Autoencoder (CAE-NN) + Multi-Scale CNN (MS-CNN)

### Datasets
The efficiency of the proposed methodology was evaluated using two public datasets from the NLM/NIH and a private dataset containing cases from Songklanagarind Hospital.

NLM collection—Montgomery County X-ray dataset (MC)
NLM collection—Shenzhen hospital X-ray dataset (SZ)
Songklanagarind hospital dataset (SK)

---
# 11

## Análisis mejorado de imágenes médicas utilizando un algoritmo híbrido de optimización de solubilidad de gas Henry con redes neuronales apiladas AdaBoost optimizadas
## Enhanced medical image analysis using hybrid Henry gas Solubility optimization algorithm with optimized AdaBoost stacked neural networks

### Fecha 2026

https://www.sciencedirect.com/science/article/pii/S0957417425024996

### Algoritmos
Feature extraction is conducted using a stacked GRU-RNN approach that captures both spatial and temporal dependencies within medical images

The HHGSO algorithm is employed to fine-tune the hyperparameters of the GRURNN. HHGSO integrates multiple metaheuristic strategies like STOA, Jaya, OSA, and BOA within a dynamic reward-penalty mechanism to enhance global search capability and avoid local minima.

The AdaBoost ensemble strategy is applied to improve classification accuracy and address class imbalance issues

Hybrid model combine extraction, optimization and classification --> HHGSO-ASGRNet

### Datasets
In this study, three publicly available medical imaging datasets were employed to ensure thorough validation and performance assessment of the proposed model across different clinical scenarios.

This Brain Tumor MRI dataset (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) comprises 7022 MRI images of the human brain

The chest X-ray (Pneumonia) dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) consists of radio-graphic images collected from pediatric patients aged between one and five years, sourced from the Women and Children’s Medical Center in Guangzhou, China. This dataset comprises a total of 5,836 chest X-ray images

The IQ-OTH/NCCD–Lung Cancer Dataset (https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) is a pub-licly available collection of CT scan images specifically utilized for lung cancer detection and classification tasks. This dataset contains a total of 1,190 CT images collected

---
# 12

## Segmentación y clasificación de tumores cerebrales: Un enfoque híbrido de aprendizaje profundo CVAE-UNETR-ResNet50-VGG16
## Brain tumor segmentation and classification: A CVAE-UNETR-ResNet50-VGG16 hybrid deep learning approach

### Fecha 2026

https://www.sciencedirect.com/science/article/pii/S1110016826000037

### Algoritmos
We use build a hybrid DL model that uses ResNet50 with VGG16 for MRI brain image classification and UNET for segmentation

### Datasets
A limited dataset comprising only 253 samples is utilized in this paper. The dataset is openly accessible in Ref. [38] which consists of cancer or normal samples with its label. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---
# 13

## FHD-HybridNet: Un marco híbrido de aprendizaje profundo guiado por la distancia de Hellinger difusa para la clasificación robusta de tumores cerebrales multiclase
## FHD-HybridNet: A fuzzy hellinger distance-guided hybrid deep learning framework for robust multiclass brain tumor classification

### Fecha 2026

https://www.sciencedirect.com/science/article/pii/S1110016826000037

### Algoritmos
Models base of CNN
DenseNet121DenseNet121
MobileNetV1MobileNetV1
ResNet50

Ensemble 
Fuzzy Hellinger Distance (FHD)

Result a hybrid architecture FHD-HybridNet
### Datasets
This study utilized a curated collection of MRI scans from multiple renowned sources rather than relying on isolated datasets. The dataset includes images from the SARTAJ dataset [21], the Br35H dataset [22], and samples provided in the study by Cheng et al., hosted on Figshare [23]. The curated dataset [24] comprises four distinct classes of MRI brain scans.

---
# 14

## MammXAI: Un enfoque de aprendizaje profundo multi-modelo adaptativo integrado con XAI para la detección de cáncer de mama usando imágenes de multimodalidad
## MammXAI: An XAI integrated adaptive multi-model deep learning approach for breast cancer detection using multi-modality images

### Fecha 2026

https://www.sciencedirect.com/science/article/pii/S1746809425016842

### Algoritmos
The proposed ETCapsNet introduces a hybrid deep learning architecture that synergistically integrates three robust components: EfficientNetv2 Small, Transformer blocks, and Capsule Networks.

XAI
Grad-CAM
Grad-CAM++
Score-CAM
SmoothGrad
Integrated Gradients
Occlusion Sensitivity
PDA (Prediction Difference Analysis)
LIME

### Datasets
The first dataset used in this study was introduced at the 15th International Conference on Image Analysis and Recognition (ICIAR-2018) as a grand challenge on BreAst Cancer Histology (BACH) images [58].

Another dataset included in this study is Breast UltraSound Images (BUSI), which were gathered in 2018 from women aged 25 to 75 [59].

The third dataset combines mammography images from three different datasets. Initially, it included 106 mass images from the InBreast dataset, 53 from the Mammographic Image Analysis Society (MIAS) dataset, and 2188 from the Digital Database for Screening Mammography (DDSM) dataset, which were split into two classes: benign and malignant [60].

---
# 15

## Mejora de la detección temprana de la enfermedad de Alzheimer mediante la arquitectura de aprendizaje automático Vision Transformer utilizando imágenes de resonancia magnética
## Enhancing Early Detection of Alzheimer’s Disease via Vision Transformer Machine Learning Architecture Using MRI Images

### Fecha 2026

https://www.mdpi.com/2078-2489/17/2/163

### Algoritmos
ViT (Vision Transformer)

ViT used Multi-Head Self-Attention, MLP, AdamW

### Datasets
In this experiment, the AD MRI dataset from Kaggle was used [38]. The 6400 pre processed MRI scans are divided into four categories: ND, MD, MoD, and VMD. 

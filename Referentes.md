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
---

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
---

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
---

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
--- 

## Un Marco Híbrido de Aprendizaje Profundo para el Diagnóstico Automatizado de Trastornos Dentales a partir de Imágenes de Rayos X
## A Hybrid Deep Learning Framework for Automated Dental Disorder Diagnosis from X-Ray Images

### Fecha 2026

https://digitalmanuscriptpedia.com/conferences/index.php/DMP-LNMR/article/download/95/95v

### Algoritmos
It combined HOG as handcrafted descriptors with DenseNet-201 and the Swin Transformer for transformer-based features, capturing complementary information and encompassing fine-grained low-level spatial characteristics as well as rich high-level semantic representations. 

### Datasets
The DRAD dataset focused on dental radiography analysis and diagnosis. The dataset consists of 1272 X-ray images. (No estoy seguro que sea publico)


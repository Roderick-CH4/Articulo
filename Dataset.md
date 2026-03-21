## Data Set

We decide use next data set https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1?resource=download

Due to the heterogeneity of the datasets used in the related works, a single public dataset (the Brain Tumor MRI Dataset) was selected to ensure a fair comparison among the models. This dataset was chosen for its adequate size, multiclass nature, and relevance in recent literature on medical image classification.

### What is a brain tumor?
A brain tumor is a collection, or mass, of abnormal cells in your brain. Your skull, which encloses your brain, is very rigid. Any growth inside such a restricted space can cause problems. Brain tumors can be cancerous (malignant) or noncancerous (benign). When benign or malignant tumors grow, they can cause the pressure inside your skull to increase. This can cause brain damage, and it can be life-threatening.

### The importance of the subject
Early detection and classification of brain tumors is an important research domain in the field of medical imaging and accordingly helps in selecting the most convenient treatment method to save patients life therefore

### About Dataset
This dataset is a combination of the following three datasets :
figshare
SARTAJ dataset
Br35H

This dataset contains 7023 images of human brain MRI images which are classified into 4 classes: glioma - meningioma - no tumor and pituitary.

no tumor class images were taken from the Br35H dataset.

I think SARTAJ dataset has a problem that the glioma class images are not categorized correctly, I realized this from the results of other people's work as well as the different models I trained, which is why I deleted the images in this folder and used the images on the figshare site instead.

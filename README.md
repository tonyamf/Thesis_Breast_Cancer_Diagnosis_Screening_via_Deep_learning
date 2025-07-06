# Breast Cancer Diagnosis using Deep Learning

This project explores the use of deep learning models for the diagnosis of breast cancer from mammography images. The primary goal was to develop a pipeline for tumor segmentation and classification using state-of-the-art architectures.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
- [Results](#results)
- [Challenges and Limitations](#challenges-and-limitations)
- [Future Work](#future-work)
- [Setup and Usage](#setup-and-usage)

## Project Overview

The project investigates a two-stage approach for breast cancer diagnosis:

1.  **Tumor Segmentation:** An EfficientNet-B0 U-Net model was designed to segment tumors from mammography images.
2.  **Tumor Classification:** A classifier was intended to take the segmented tumor as input and diagnose it as benign or malignant.

Due to the segmentation model's failure to produce accurate results, an alternative approach of direct image classification was also explored.

## Dataset

The primary dataset used in this project is a subset of a digital mammography dataset. The data includes images with corresponding tumor contours, which were used to create masks for the segmentation task. The dataset is characterized by:

-   Limited number of samples
-   Class imbalance (normal, benign, malignant)
-   Variations in image quality and tumor appearance

The Oxford Flowers dataset was also used to validate the classification model's performance on a standard benchmark.

## Methodology

### Segmentation

-   **Model:** An EfficientNet-B0 encoder was integrated with a U-Net decoder to create an EfficientNet-B0-U-Net model. The pretrained EfficientNet-B0 was intended to provide robust feature extraction.
-   **Data Preparation:** Masks were generated from the tumor contours provided in the dataset. The images and masks were preprocessed and augmented to increase the training data size.
-   **Outcome:** The segmentation model failed to learn the task effectively, producing a Dice loss of only 1% after the first epoch. This was attributed to the small dataset size and the complexity of the task.

### Classification

-   **Model:** An EfficientNet-B0 model was used for direct image classification.
-   **Data Preparation:** Images were categorized into three classes: normal, benign, and cancer. The model was also tested on a two-class problem (benign vs. cancer).
-   **Outcome:** The classification model also failed to converge on the mammography data, achieving only 47% accuracy for the three-class problem and 52% for the two-class problem. However, the same model achieved 100% accuracy on the Oxford Flowers dataset, indicating that the issue was data-related rather than a flaw in the model architecture.

## Results

The experimental results were not successful in achieving the primary objectives. The key outcomes were:

-   **Segmentation:** The model failed to learn the segmentation task.
-   **Classification:** The model failed to learn the classification task on the mammography data but performed exceptionally well on a standard dataset.

These results highlight the challenges of working with limited and complex medical imaging data.

## Challenges and Limitations

-   **Data Scarcity:** The primary limitation was the small size of the dataset, which was insufficient for training deep learning models from scratch or for effective fine-tuning.
-   **Class Imbalance:** The dataset had a significant imbalance between the different classes, which can bias the model towards the majority class.
-   **Data Quality:** The mammography images had variations in quality, and the tumor contours were not always precise, which may have affected the mask generation process.
-   **Model Complexity:** The chosen models, while powerful, may have been too complex for the small dataset, leading to overfitting or failure to converge.

## Future Work

-   **Data Augmentation:** Implement more advanced data augmentation techniques, such as Generative Adversarial Networks (GANs), to generate synthetic data.
-   **Transfer Learning:** Utilize models pretrained on larger medical imaging datasets to leverage domain-specific features.
-   **Alternative Architectures:** Explore lighter-weight models that are more suitable for small datasets.
-   **Class Imbalance Strategies:** Employ techniques such as oversampling, undersampling, or focal loss to address the class imbalance issue.

## Setup and Usage

The project is implemented in Jupyter Notebooks, and there's a bit a prompt coding correlated with pyhon files but i didn't check them, but the whole projected was made in 2021 with those Jupyter Notebooks in 2021 and it works and the paper explain it. The ai prompt code are alternative solution, maybe it will be easier to understand but I didn't check the python generated code or run them to know. To run the code, you will need to:

1.  Set up a Python environment with the required libraries (TensorFlow, Keras, etc.).
2.  Mount your Google Drive to access the dataset.
3.  Run the cells in the notebook sequentially to reproduce the experiments.

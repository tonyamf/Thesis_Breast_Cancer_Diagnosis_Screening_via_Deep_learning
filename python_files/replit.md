# Breast Cancer Diagnosis via Deep Learning

## Overview

This repository implements a comprehensive deep learning pipeline for automated breast cancer detection on screening mammography images. The system combines multiple state-of-the-art architectures including U-Net for segmentation, EfficientNet for classification, and Meta Pseudo Labels for semi-supervised learning. The application is built as an interactive Streamlit web interface that allows users to analyze mammography images, train models, and evaluate performance.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Interface**: Multi-page dashboard with navigation sidebar
- **Visualization**: Integrated Plotly and Matplotlib for interactive charts and medical image display
- **Deployment**: Configured for Replit autoscale deployment on port 5000

### Backend Architecture
- **Core Framework**: TensorFlow/Keras for deep learning models
- **Image Processing**: OpenCV and scikit-image for medical image preprocessing
- **Data Augmentation**: Albumentations library for advanced augmentation techniques
- **Model Architecture**: Modular design with separate model classes for different tasks

### Model Components
1. **U-Net Segmentation**: Medical image segmentation for region of interest detection
2. **EfficientNet Classification**: Multi-class classification (Normal, Benign, Malignant)
3. **Meta Pseudo Labels**: Semi-supervised learning framework for leveraging unlabeled data

## Key Components

### Models (`/models/`)
- **UNet**: Implements U-Net architecture for medical image segmentation with customizable depth and filters
- **EfficientNet**: Wrapper for EfficientNet B0-B7 variants with transfer learning capabilities
- **Meta Pseudo Labels**: Teacher-student framework for semi-supervised learning with confidence thresholding

### Utilities (`/utils/`)
- **Preprocessing**: DICOM tag removal, artifact cleaning, contrast enhancement (CLAHE)
- **Augmentation**: Medical image-specific augmentations including geometric and intensity transformations
- **Metrics**: Comprehensive evaluation metrics for both classification and segmentation tasks
- **Visualization**: Advanced plotting utilities for training curves, confusion matrices, and medical image overlays

### Training Pipeline (`/training/`)
- **Classification Trainer**: Complete training pipeline for EfficientNet-based classification
- **Segmentation Trainer**: U-Net training with specialized medical image loss functions
- **MPL Trainer**: Meta Pseudo Labels training coordinator for semi-supervised learning

## Data Flow

1. **Input Processing**: Raw mammography images → DICOM tag removal → artifact cleaning → normalization
2. **Augmentation**: Preprocessed images → geometric/intensity augmentations → training-ready dataset
3. **Model Training**: Augmented data → model-specific training pipelines → trained models
4. **Inference**: New images → preprocessing → model prediction → confidence scoring
5. **Visualization**: Results → interactive dashboards → performance metrics and medical image overlays

## External Dependencies

### Core ML/Scientific Computing
- **TensorFlow 2.14+**: Deep learning framework for model implementation and training
- **NumPy 2.3+**: Numerical computing foundation
- **SciPy 1.16+**: Scientific computing utilities
- **scikit-learn 1.7+**: Machine learning utilities and metrics
- **scikit-image 0.25+**: Image processing algorithms

### Image Processing
- **OpenCV 4.11+**: Computer vision and image processing
- **Albumentations 2.0+**: Advanced data augmentation library
- **PIL/Pillow**: Image format handling

### Visualization and Interface
- **Streamlit 1.46+**: Web application framework
- **Matplotlib 3.10+**: Static plotting
- **Plotly 6.1+**: Interactive visualizations
- **Seaborn 0.13+**: Statistical data visualization
- **Pandas 2.3+**: Data manipulation and analysis

### System Dependencies (via Nix)
- **Cairo, FreeType**: Graphics rendering support
- **FFmpeg**: Video/image codec support
- **GTK3, OpenGL**: GUI framework support
- **Ghostscript**: PostScript/PDF processing

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Nix package management
- **Runtime**: Python 3.11 with scientific computing stack
- **Package Manager**: UV for Python dependency resolution

### Production Deployment
- **Target**: Replit autoscale deployment
- **Port**: 5000 (configured for external access)
- **Process Management**: Streamlit server with automatic restart capabilities
- **Scaling**: Automatic scaling based on demand

### Model Persistence
- **Format**: TensorFlow SavedModel format for deployment
- **Storage**: Local filesystem with configurable model directories
- **Versioning**: Timestamp-based model versioning system

## Changelog

Changelog:
- June 23, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.
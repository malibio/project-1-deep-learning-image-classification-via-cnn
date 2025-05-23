# Deep Learning Image Classification via CNN

## Team Members
- **Alejandro Zahinos**
- **Baggiyam Shanmugam** 
- **Michael Libio**

## Project Overview
This repository contains our exploration of Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset. We systematically developed and compared multiple CNN architectures to understand the impact of different design choices on model performance.

## Dataset
- **CIFAR-10**: 60,000 32×32 color images across 10 classes
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Split**: 50,000 training, 10,000 test images

## Model Architectures Explored

### 1. Simple Baseline CNN (`CNN Model 1`)
- Basic 2-layer architecture (32→64 filters)
- **Accuracy**: 70.74%
- Minimal regularization, no data augmentation

### 2. ResNet-Inspired CNN (`CNN Model 2`) 
- Skip connections and modern regularization
- **Accuracy**: 82.65%
- BatchNormalization, progressive dropout, GlobalAveragePooling

### 3. Modern VGG-Style CNN (`CNN Model 3`)
- Paired convolutions with consistent regularization
- **Accuracy**: 84.46%
- Most parameter-efficient model (307K parameters)

### 4. Deep CNN with Heavy Regularization (`CNN Model 4`)
- Deeper architecture (64→128→256 filters)
- **Accuracy**: 89.10%
- Comprehensive augmentation and regularization

### 5. Model Deployment (`deployment.ipynb`)
- **Gradio web application** for real-time CIFAR-10 image classification
- User-friendly drag-and-drop interface for uploading images
- **Live predictions** with confidence scores for all 10 classes (top 3 displayed)
- **Public sharing capability** with shareable URLs for easy access
- Loads the best-performing VGG-style model (89.10% accuracy)

## Key Technical Components

### Data Pipeline (`cnn_utils.py`)
- Automated CIFAR-10 download and preprocessing
- Consistent train/validation/test splits
- Standardized evaluation metrics and visualizations

### Data Augmentation Strategies
- **Basic**: Rotation (±15°), shifts (10%), horizontal flip
- **Enhanced**: Rotation (±20°), shifts (15%), brightness variation, channel shifts

### Training Optimizations
- Early stopping and learning rate scheduling
- Progressive dropout and spatial dropout
- L2 weight regularization
- Two-phase transfer learning (frozen → fine-tuning)

### Model Deployment
- **Gradio interface** for intuitive image upload and classification
- Real-time inference with confidence scoring
- Public URL sharing for remote access
- Automatic image preprocessing (resize to 32×32, normalization)

## Results Summary

| Model | Architecture | Parameters | Accuracy | Key Innovation |
|-------|-------------|------------|----------|----------------|
| Model 1 | Simple CNN | 545K | 70.74% | Baseline |
| Model 2 | ResNet-inspired | 319K | 82.65% | Skip connections |
| Model 3 | VGG-style | 307K | 84.46% | Parameter efficiency |
| Model 4 | Deep VGG | 3.25M | 89.10% | Heavy regularization |

## Repository Structure
```
├── cnn_utils.py                          # Shared utilities and pipeline
├── vgg_style_with_batchnorm.ipynb       # Main VGG-style implementation
├── transfer_learning.ipynb              # Transfer learning experiments
├── deployment.ipynb                     # Gradio web app for model deployment
├── model_results/                       # Performance analysis documents
└── README.md                           # Project documentation
```

## Key Insights
1. **Architecture matters**: Skip connections and paired convolutions significantly improve performance
2. **Regularization is crucial**: BatchNormalization and dropout prevent overfitting
3. **Data augmentation impact**: Even mild augmentation provides 12-14% accuracy gains
4. **Parameter efficiency**: Model 3 achieves 84% accuracy with only 307K parameters
5. **Transfer learning potential**: Pre-trained models can exceed 90% accuracy with proper fine-tuning

## Usage
```python
import cnn_utils

# Load and preprocess data
data_dict = cnn_utils.load_cifar10_from_tar()
data = cnn_utils.preprocess_data(data_dict)

# Train any model architecture
model = create_your_model()
history = cnn_utils.train_model(model, data, augmentation=augmentation)
cnn_utils.evaluate_model(model, data, history)
```

## Technologies Used
- **TensorFlow/Keras**: Deep learning framework
- **Python**: Data processing and visualization
- **scikit-learn**: Evaluation metrics
- **matplotlib/seaborn**: Result visualization

---
*This project demonstrates systematic CNN architecture exploration, achieving state-of-the-art results on CIFAR-10 through careful design choices and modern deep learning practices.*
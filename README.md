# Emotion Detection Using CNN

A deep learning project for emotion recognition using Convolutional Neural Networks (CNN). This repository implements emotion detection models trained on multiple datasets including AffectNet and FER-2013, utilizing transfer learning with state-of-the-art architectures.

## 📋 Overview

This project aims to classify human facial expressions into different emotion categories using deep learning techniques. The implementation includes multiple model architectures and experiments across different datasets to achieve optimal emotion recognition performance.

## 🎯 Emotion Categories

The models can detect the following emotions:
- 😡 **Anger**
- 🤢 **Disgust**
- 😱 **Fear**
- 😊 **Happy**
- 😐 **Neutral**
- 😔 **Sadness**
- 😲 **Surprise**

## 🗂️ Repository Structure

```
Emotion-Detection-Using-CNN/
├── AffectNet/
│   └── DenseNet 1.ipynb          # DenseNet implementation on AffectNet dataset
├── FER-2013/
│   └── Resnet.ipynb              # ResNet implementation on FER-2013 dataset
├── Other/
│   └── DenseNet 1.ipynb          # Alternative DenseNet experiments
└── README.md
```

## 📊 Datasets

### 1. AffectNet Dataset
- Large-scale facial expression dataset
- Contains multiple emotion categories
- Image size: 48x48 pixels
- Dataset source: [Kaggle - AffectNet](https://www.kaggle.com/datasets/thienkhonghoc/affectnet)
- Size: ~1.75GB

### 2. FER-2013 Dataset
- Facial Expression Recognition 2013 dataset
- 7 emotion categories
- Grayscale images of 48x48 pixels
- Dataset source: [Kaggle - Emotion Detection FER](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)
- Size: ~65MB

## 🏗️ Model Architectures

### DenseNet (Dense Convolutional Network)
- **Implementation**: Transfer learning with pre-trained DenseNet
- **Dataset**: AffectNet
- **Features**:
  - Global Average Pooling
  - Multiple Dense layers with dropout regularization
  - L2 regularization to prevent overfitting
  - Custom classifier head with 8 output classes

**Architecture Details**:
```
- Base Model: DenseNet (pre-trained)
- Global Average Pooling
- Dense Layer (256 units) + Dropout (0.3)
- Dense Layer (1024 units) + Dropout (0.5)
- Dense Layer (512 units) + Dropout (0.5)
- Output Layer (8 classes, softmax)
```

### ResNet (Residual Network)
- **Implementation**: Transfer learning with ResNet
- **Dataset**: FER-2013
- **Features**: Residual connections for deeper network training

## 🛠️ Technologies & Libraries

- **Deep Learning Framework**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Evaluation**: Scikit-learn
- **Development Environment**: Google Colab (GPU: T4)

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib seaborn plotly scikit-learn kaggle
```

### Dataset Setup

1. **Set up Kaggle API credentials**:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
```

2. **Download AffectNet dataset**:
```bash
kaggle datasets download -d thienkhonghoc/affectnet
```

3. **Download FER-2013 dataset**:
```bash
kaggle datasets download -d ananthu017/emotion-detection-fer
```

### Training Configuration

**Hyperparameters**:
- Image Size: 48x48 pixels
- Batch Size: 64
- Initial Epochs: 30
- Fine-tuning Epochs: 20
- Learning Rate: 0.01
- Early Stopping Patience: 3 epochs

## 📈 Training Process

### Two-Stage Training:

1. **Feature Extraction Stage**:
   - Freeze pre-trained base model layers
   - Train only the custom classifier head
   - Learn emotion-specific features

2. **Fine-tuning Stage**:
   - Unfreeze base model layers
   - Lower learning rate (0.001)
   - Fine-tune entire network

### Data Augmentation

The models use `ImageDataGenerator` with:
- Rescaling: 1./255 (normalization)
- Random shuffling
- Batch processing

## 📊 Model Evaluation

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance analysis
- **Classification Report**: Precision, recall, F1-score
- **ROC-AUC Score**: Multi-class ROC analysis

## 💾 Model Export

Trained models are saved in HDF5 format:
```python
model.save("emotion_detection_model.h5")
```

## 🔬 Experiments

### AffectNet Experiments
- **Location**: `AffectNet/DenseNet 1.ipynb`
- **Model**: DenseNet with custom classifier
- **Classes**: 8 emotion categories (extended from 7)
- **Optimization**: SGD optimizer with momentum

### FER-2013 Experiments
- **Location**: `FER-2013/Resnet.ipynb`
- **Model**: ResNet architecture
- **Classes**: 7 emotion categories
- **Dataset**: Standard FER-2013

### Additional Experiments
- **Location**: `Other/DenseNet 1.ipynb`
- **Purpose**: Alternative configurations and hyperparameter tuning

## 📝 Usage Example

```python
# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('emotion_detection_model.h5')

# Preprocess input image
import tensorflow as tf
preprocess_input = tf.keras.applications.densenet.preprocess_input

# Make prediction
predictions = model.predict(preprocessed_image)
emotion_class = CLASS_LABELS[np.argmax(predictions)]
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- **Datasets**: 
  - AffectNet dataset creators
  - FER-2013 challenge organizers
  - Kaggle dataset contributors
- **Frameworks**: TensorFlow/Keras team
- **Pre-trained Models**: ImageNet pre-trained weights

## 📧 Contact

For questions or collaborations, please open an issue in this repository.

---

**Note**: This project is designed for research and educational purposes in the field of affective computing and emotion recognition.
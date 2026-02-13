# MNIST Handwritten Digit Classification using CNN

A deep learning project that classifies handwritten digits (0-9) using Convolutional Neural Networks (CNN) built with TensorFlow, achieving **98.98% accuracy** on the MNIST dataset.

## Overview

This project implements a Convolutional Neural Network to recognize handwritten digits from the famous MNIST dataset. The model learns to identify patterns in pixel data and accurately classify digits through multiple convolutional and pooling layers.

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**
- **Training Set**: 60,000 grayscale images
- **Test Set**: 10,000 grayscale images
- **Image Size**: 28×28 pixels
- **Classes**: 10 (digits 0-9)
- **Format**: Grayscale values (0-255)

The MNIST dataset is one of the most well-known benchmarks in machine learning and computer vision.

## Model Architecture

### Convolutional Layers
```
Input Layer (28×28×1)
    ↓
Conv2D (32 filters, 5×5 kernel)
    ↓
MaxPooling2D (2×2)
    ↓
Dropout (regularization)
    ↓
Conv2D (64 filters, 3×3 kernel)
    ↓
MaxPooling2D (2×2)
    ↓
Dropout (regularization)
```

### Fully Connected Layers
```
Flatten
    ↓
Dense (128 neurons, ReLU activation)
    ↓
Dropout (regularization)
    ↓
Output Dense (10 neurons, Softmax activation)
```

### Architecture Details

**Convolutional Layers:**
- **Layer 1**: 32 filters (5×5) - Extracts basic features like edges
- **Layer 2**: 64 filters (3×3) - Learns complex patterns

**Pooling Layers:**
- 2×2 MaxPooling - Reduces spatial dimensions while retaining important features

**Regularization:**
- Dropout layers - Prevents overfitting by randomly dropping neurons during training

**Fully Connected:**
- 128-neuron hidden layer with ReLU activation
- 10-neuron output layer with Softmax for probability distribution

## Performance

- **Test Accuracy**: 98.98%
- **Evaluation**: Confusion matrix visualization using Seaborn heatmap

The model demonstrates excellent performance in correctly classifying handwritten digits across all 10 classes.

## Project Structure

```
mnist-cnn-classifier/
├── MNIST-CNN__1_.ipynb          # Main notebook
├── README.md                    # This file
└── requirements.txt             # Dependencies
```

## Requirements

```txt
tensorflow>=2.0
numpy
matplotlib
seaborn
jupyter
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-cnn-classifier.git
cd mnist-cnn-classifier

# Install dependencies
pip install tensorflow numpy matplotlib seaborn jupyter
```

### Run the Notebook

```bash
jupyter notebook MNIST-CNN__1_.ipynb
```

### Training the Model

The notebook automatically:
1. Loads the MNIST dataset from TensorFlow/Keras
2. Preprocesses and normalizes the images
3. Builds the CNN architecture
4. Trains the model
5. Evaluates performance
6. Visualizes results with confusion matrix

## Key Features

✅ **Data Preprocessing**: Normalization and reshaping of images  
✅ **Visual Inspection**: Display sample images from dataset  
✅ **CNN Architecture**: Multi-layer convolutional network  
✅ **Regularization**: Dropout layers to prevent overfitting  
✅ **Performance Evaluation**: Confusion matrix visualization  
✅ **High Accuracy**: 98.98% test accuracy  

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualizations and heatmaps
- **Jupyter Notebook**: Interactive development environment

## How CNNs Work for Image Classification

### Why CNNs?

Unlike traditional neural networks, CNNs are designed specifically for image data:

1. **Spatial Hierarchy**: Learns features from simple (edges) to complex (shapes)
2. **Parameter Sharing**: Same filter applied across the image reduces parameters
3. **Translation Invariance**: Recognizes patterns regardless of position
4. **Local Connectivity**: Each neuron only connects to a small region

### Training Process

1. **Forward Pass**: Input image → Convolutions → Pooling → Fully Connected → Output
2. **Loss Calculation**: Compares predictions with true labels
3. **Backpropagation**: Updates weights to minimize loss
4. **Optimization**: Adam optimizer adjusts learning rate

## Results Visualization

The project includes:
- Sample images from the dataset
- Training/validation accuracy curves
- Confusion matrix heatmap showing prediction performance across all digits

## Model Performance Analysis

The confusion matrix reveals:
- Strong performance across all digit classes
- Minimal misclassifications
- Potential confusion between similar-looking digits (e.g., 4 and 9)

## Future Enhancements

- [ ] Data augmentation (rotation, scaling, shifting)
- [ ] Deeper architectures (ResNet, VGG-style)
- [ ] Batch normalization layers
- [ ] Learning rate scheduling
- [ ] Model ensemble techniques
- [ ] Real-time digit recognition from webcam
- [ ] Deployment as web application (Flask/Streamlit)
- [ ] Transfer learning experiments
- [ ] Testing on different handwritten digit datasets

## Applications

This type of model can be applied to:
- Check processing in banking
- Postal code recognition
- Form digitization
- Historical document analysis
- Educational tools for learning
- Captcha verification systems

## Learning Outcomes

This project demonstrates:
- Building CNNs from scratch with TensorFlow
- Understanding convolutional and pooling operations
- Implementing regularization techniques
- Evaluating classification models
- Visualizing deep learning results

## Contributing

Contributions are welcome! Feel free to:
- Improve model architecture
- Add data augmentation
- Enhance visualizations
- Report issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yann LeCun et al. for the MNIST dataset
- TensorFlow and Keras teams
- Computer vision research community

## Author

**Harshith Raj P**

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating fundamental concepts in deep learning and computer vision. 🎯

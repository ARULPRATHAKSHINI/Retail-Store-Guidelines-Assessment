# Retail Store Compliance Classification

A computer vision system for automatically classifying retail store environments as "compliant" (well-organized) or "non-compliant" (cluttered) using deep learning and providing visual explanations.

## Project Overview

This project uses transfer learning with MobileNetV2 to analyze retail store images and determine their compliance with organization standards. The system not only classifies images but also provides visual explanations using Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight areas of concern or compliance.


## Features

- **Automated Classification**: Binary classification of retail spaces as compliant or non-compliant
- **Visual Explanations**: Heat map generation to highlight areas influencing the decision
- **Transfer Learning**: MobileNetV2 architecture with pre-trained weights for efficient training
- **Data Augmentation**: Rotation, zoom, shift, and flip transformations to improve model generalization
- **Extensible Architecture**: Modular design for easy enhancement and modification

## Directory Structure

```
retail-compliance/
├── data.py                # Image downloading script
├── organize.py            # Dataset organization 
├── processing.py          # Data preprocessing and generator creation
├── advance.py             # Model architecture definition
├── train.py               # Model training and evaluation
├── grad_cam.py            # Gradient-based visualization implementation
├── visualization.py       # Utility functions for visualization
├── predict.py             # Script for inference on new images
├── models/                # Directory for saved models
│   ├── best_model.h5      # Best model saved during training
│   ├── final_model.h5     # Final model after training completion
│   ├── class_indices.json # Class mapping information
│   └── training_history.png # Training metrics visualization
├── dataset/               # Training dataset
│   ├── compliant/         # Well-organized store images
│   └── non-compliant/     # Cluttered store images
├── simple_images/         # Raw downloaded images
│   ├── well_organized_retail_store_interior/
│   └── cluttered_retail_store_interior/
└── results/               # Directory for inference results
    └── *.png              # Analysis visualizations
```

## Requirements

### Hardware Requirements
- Multi-core processor (Intel Core i5/AMD Ryzen 5 or better)
- 8GB RAM (16GB recommended)
- NVIDIA GPU with CUDA support (recommended for training)

### Software Requirements
- Python 3.6+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- simple_image_download

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/retail-compliance.git
   cd retail-compliance
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install tensorflow opencv-python numpy matplotlib simple_image_download pillow
   ```

## Usage

### Data Collection

To download images for training:

```
python data.py
```

This will download images of organized and cluttered retail stores using simple_image_download.

### Dataset Organization

To organize downloaded images into the correct dataset structure:

```
python organize.py
```

This creates a structured dataset with separate folders for compliant and non-compliant images.

### Model Training

To train the classification model:

```
python train.py
```

This will:
1. Create data generators with augmentation
2. Build and compile the MobileNetV2 model with custom classification head
3. Train for up to 20 epochs with early stopping
4. Save the best model to `models/best_model.h5`
5. Generate training history visualization

### Making Predictions

To analyze a new retail store image:

```
python predict.py
```

Note: By default, the script looks for an image at the hardcoded path in predict.py. You'll need to modify this path to point to your test image:

```python
# In predict.py
img_path = r'path/to/your/test/image.jpg'
```

### Output

The prediction script will:
1. Load the trained model
2. Process the test image
3. Generate a classification (compliant/non-compliant)
4. Create a Grad-CAM visualization
5. Save the analysis image to the `results` directory
6. Print the classification result with confidence score

## Model Architecture

This project uses MobileNetV2 with transfer learning:

- Base model: MobileNetV2 (pre-trained on ImageNet)
- Custom classification head:
  - Global Average Pooling
  - Dense layer (128 neurons, ReLU activation)
  - Output layer (1 neuron, Sigmoid activation)
- Optimizer: Adam
- Loss function: Binary Cross-Entropy

## Visualization Technique

The system uses Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps that highlight regions of the image most influential to the classification decision. This helps users understand why a particular store was classified as compliant or non-compliant.

## Customization

### Using Your Own Dataset

To use your own images:
1. Place compliant images in `dataset/compliant/`
2. Place non-compliant images in `dataset/non-compliant/`
3. Run `python train.py` to train on your dataset

### Modifying Model Architecture

To change the model architecture, edit the `create_advanced_model()` function in `advance.py`.

### Adjusting Training Parameters

Training parameters can be modified in `train.py`:
```python
# Change number of epochs
epochs = 20  # Default is 20

# Change batch size in processing.py
batch_size = 32  # Default is 32

# Adjust early stopping patience
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Default is 5
    verbose=1
)
```

## Troubleshooting

### Common Issues

1. **"Model file not found" error**:
   - Make sure you've run `train.py` before `predict.py`
   - Check if the model file exists at `models/best_model.h5`

2. **"Image file not found" error**:
   - Update the image path in `predict.py` to point to your test image

3. **Low accuracy during training**:
   - Ensure your dataset has sufficient images in both classes
   - Try increasing the number of training epochs
   - Adjust data augmentation parameters in `processing.py`

4. **Grad-CAM visualization issues**:
   - If visualization fails, check that the convolutional layer name exists in your model
   - The system will fall back to the last convolutional layer if the specified one isn't found

## Future Enhancements

- Multi-class classification (excellent, good, needs improvement, poor)
- Object detection integration to identify specific organizational issues
- Web service or mobile application development
- Video analysis capabilities

## Contributors

[Your Name]

## License

[Specify your license here]

## Acknowledgments

- The MobileNetV2 architecture and pre-trained weights from TensorFlow/Keras
- Grad-CAM implementation inspired by [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

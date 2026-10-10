# Intelligent Industrial Defect Detection and Classification System

This project focuses on the automated detection and multi-class classification of surface defects in industrial products. It supports various mainstream deep learning model architectures and is suitable for industrial quality inspection and defect localization scenarios.

## Features
- Supports multi-class classification models (12 classes: defect-free + 11 defect types)
- Supports three backbone networks: SimpleCNN (custom CNN), MobileNetV2, EfficientNetB0
- Supports test result visualization (accuracy, confusion matrix, sample distribution, etc.)
- Supports exporting detection results to Excel with image previews

## Directory Structure
```
├── models/                # Model definitions
│   └── quality_control_cnn.py      # Includes SimpleCNN, MobileNetV2, EfficientNetB0, and factory method
├── datasets/              # Dataset processing
│   └── neu_dataset.py     # Dataset loading and preprocessing
├── train/                 # Training scripts
│   ├── binary_train.py    # Binary classification training script
│   ├── multi_train.py     # Multi-class classification training script
├── test/                  # Testing and visualization scripts
│   ├── binary_test.py            # Binary classification accuracy test
│   ├── multiclass_accuracy.py    # Multi-class accuracy test (with charts)
│   └── multiclass_count.py       # Multi-class statistics and Excel export
├── checkpoints/           # Trained model weights
├── utils/                 # Utility functions
├── config.py              # Configuration file (paths, parameters, etc.)
├── common/                # Common enums, utilities, etc.
│   └── enums.py           # Class names, etc.
├── data/                  # Dataset directory
│   └── binary/            # Binary classification dataset
│   └── multi/             # Multi-class dataset
│   └── test/              # Test dataset
├── requirements.txt       # Dependency list
└── README.md              # Project documentation
```

## Supported Model Architectures
- `SimpleCNN`: Custom lightweight convolutional neural network
- `MobileNetV2Classifier`: Efficient and lightweight, suitable for speed-critical scenarios
- `EfficientNetB0Classifier`: Efficient and high-accuracy
- Easily switch models via the `get_model(model_name, num_classes=12, pretrained=True)` factory method

## Defect Classes
- 0 - Normal (No defect)
- 1 - Non-conductive
- 2 - Convex powder
- 3 - rub The Flowers
- 4 - Orange peel
- 5 - Concave
- 6 - Cracking Coating
- 7 - Leaky Bottom
- 8 - Bump
- 9 - Dirty Spots
- 10 - Pitting
- 11 - Other

## Dataset Format
- **Classification task**: Images are organized into folders by class
- **Detection task**: Images are organized into folders by class

## Main Dependencies
- torch, torchvision
- numpy, pandas, matplotlib, seaborn
- opencv-python
- scikit-learn
- openpyxl

## Recommended Environment
- Python 3.8+
- GPU acceleration is recommended

## Install Dependencies
```
pip install -r requirements.txt
```

## Data Preparation
- Download the dataset and extract it to the `data/` directory, with the following structure:
  ```
  data/
    binary/
    multi/
    test/
  ```
  `defective` is for defect samples, `normal` is for defect-free samples.

## Training and Testing
### Training (Multi-class example)
```sh
python train/multi_train.py
```

### Testing and Visualization
```sh
python test/multiclass_accuracy.py
```
- Outputs overall accuracy, per-class accuracy, sample distribution charts, etc.

### Export Detection Results to Excel
```sh
python test/multiclass_count.py
```
- Automatically generates `results.xlsx` with image previews and classification results

### Single-image defect checking
```sh
python cnn.py
# args
## --class_type: binary, multi
## --img_path: picture path
## --model_name: quality, mobilenetv2, efficientnetb0
```
- Generate predictions

### YOLO Single-image defect checking
```sh
python yolo.py
# args
## --img_path: picture path
```
- Generate predictions

## Contact & Support
For questions or customization needs, please contact the project maintainer. 
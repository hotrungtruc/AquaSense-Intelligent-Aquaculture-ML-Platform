# AquaSense: Intelligent Aquaculture ML Platform

A comprehensive machine learning platform for aquaculture monitoring and optimization using computer vision, audio analysis, and data forecasting. Built with PyTorch, TensorFlow, and YOLOv8.

## 📋 Overview

AquaSense provides an end-to-end ML solution for aquaculture operations, enabling intelligent monitoring, optimization, and decision support through multiple AI modules:

- **Smart Feeding System**: Audio-based fish feeding behavior detection (CNN6 model)
- **Fish Tracking & Counting**: YOLOv8-based fish detection and pose estimation
- **Disease Classification**: ResNet50-based freshwater fish disease detection
- **Biomass Estimation**: Fish size and weight prediction
- **Water Quality Forecasting**: Time series prediction for aquaculture parameters

## 🏗️ Project Structure

```
AquaSense-Intelligent-Aquaculture-ML-Platform/
├── src/                               # Source code modules
│   ├── smart_feeding/                # Fish feeding behavior detection
│   │   ├── models.py                # Cnn6 model architecture
│   │   ├── fish_voice_dataset.py    # Audio dataset loader
│   │   ├── train.py                 # Training pipeline
│   │   ├── evaluate.py              # Model evaluation utilities
│   │   ├── visualize_results.py     # Inference and visualization
│   │   ├── losses.py                # Custom loss functions
│   │   ├── ultils.py                # Utility functions
│   │   ├── main.py                  # Main entry point (config-based)
│   │   └── config.py                # Hyperparameter configuration
│   │
│   ├── tracking_counting/            # Fish tracking and counting
│   │   └── Tracking_counting.ipynb   # YOLOv8 pose estimation notebook
│   │
│   ├── health_monitoring/            # Disease classification
│   │   └── fresh-water-fish-disease-classification.ipynb
│   │
│   ├── biomass_estimation/           # Biomass prediction
│   │   └── Train.ipynb
│   │
│   └── water_quality_prediction/     # Water quality forecasting
│       └── aquaponics-fish-pond-datasets-forecasting-model.ipynb
│
├── config/                            # Configuration files
│   ├── requirements.txt              # Project dependencies
│   └── yolov8_config.yaml           # YOLOv8 dataset configuration
│
├── data/                              # Datasets
│   ├── Fish_feeding_sounds/          # Audio data for smart feeding
│   │   ├── strong/                  # Strong feeding sounds
│   │   ├── middle/                  # Medium feeding sounds
│   │   ├── weak/                    # Weak feeding sounds
│   │   └── None/                    # No feeding sounds
│   │
│   ├── Freshwater_Fish_Disease/      # Fish disease dataset
│   ├── Couting_tracking_biomass/     # Tracking and biomass data
│   │   ├── fish-skeleton-yolo/      # YOLO-formatted pose estimation data
│   │   └── fish_skeleton_coco/      # COCO-formatted original data
│   │
│   └── public_datasets.md            # Links to public datasets
│
├── models/                            # Pre-trained and trained models
│   ├── biomass_estimation/
│   ├── health_monitoring/
│   ├── smart_feeding/
│   └── tracking_counting/
│
├── results/                           # Training results and outputs
│   └── smart_feeding/               # Smart feeding experiment results
│       └── <exp_name>/              # Experiment directory
│           ├── save_models/         # Saved checkpoints
│           ├── logs/                # Training logs
│           ├── plots/               # Visualization outputs
│           └── training_history.csv # Training metrics
│
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/hotrungtruc/AquaSense-Intelligent-Aquaculture-ML-Platform.git
cd AquaSense-Intelligent-Aquaculture-ML-Platform
```

2. **Install dependencies**
```bash
pip install -r config/requirements.txt
```

### Core Dependencies

- **Deep Learning**: torch, torchvision, tensorflow, keras
- **Audio Processing**: librosa, torchaudio, torchlibrosa
- **Object Detection**: ultralytics (YOLOv8)
- **Computer Vision**: opencv-python
- **Data Processing**: numpy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, pathlib

## 📚 Module Documentation

### 1. Smart Feeding System (`src/smart_feeding/`)

**Purpose**: Detect and classify fish feeding behavior from acoustic data.

**Model**: Cnn6 - A 7-layer CNN designed for spectrogram-based audio classification
- Input: 32kHz audio, 2-second clips
- Output: 4-class classification (strong, middle, weak, no feeding)
- Features: Mel-spectrogram, batch normalization, mixup augmentation

**Key Files**:
- `models.py` - Cnn6 architecture with spectrogram feature extraction
- `fish_voice_dataset.py` - Audio dataset loader with resampling and padding
- `train.py` - Training loop with per-epoch evaluation and checkpointing
- `evaluate.py` - Inference and metric computation (mAP, AUC, accuracy)
- `visualize_results.py` - Load checkpoints and generate performance plots
- `config.py` - Hyperparameter configuration (batch size, learning rate, etc.)

**Usage**:

```python
# Train a model
python src/smart_feeding/train.py

# Run inference and generate plots
python src/smart_feeding/visualize_results.py \
  --ckpt models/smart_feeding/best.pt \
  --batch-size 32 \
  --sample-rate 32000
```

**Output Structure**:
```
results/smart_feeding/<exp_name>/
├── save_models/          # Checkpoint files (.pt)
├── logs/                 # Training logs
├── plots/                # Visualization images
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.png
│   └── metrics_bar_chart.png
└── training_history.csv  # Epoch-wise metrics
```

### 2. Fish Tracking & Counting (`src/tracking_counting/`)

**Purpose**: Detect, track, and estimate pose of individual fish using YOLOv8.

**Model**: YOLOv8n-pose (nano variant for speed)
- Task: Instance segmentation + 7-keypoint pose estimation
- Classes: 2 (MrTri-descendants, generic fish)
- Keypoints: mouth, left_fin, right_fin, body1, body2, body3, tail

**Dataset Format**: YOLO format with normalized keypoint coordinates
- Data: Pre-converted COCO format → YOLO-pose format
- Location: `data/Couting_tracking_biomass/fish-skeleton-yolo/`
- Splits: train/ (images + labels), valid/, test/

**Notebook Workflow**:
1. Load pre-converted YOLO dataset
2. Train YOLOv8n-pose on custom fish dataset
3. Visualize pose predictions on test images
4. Generate pose detection video
5. Evaluate metrics (mAP50, mAP50-95 for boxes and keypoints)

**Usage**:
Open and run `src/tracking_counting/Tracking_counting.ipynb` in Jupyter

### 3. Disease Classification (`src/health_monitoring/`)

**Purpose**: Classify freshwater fish diseases from images.

**Model**: ResNet50 transfer learning
- Input: 256×256 RGB images
- Output: Multi-class disease classification
- Training: Supervised learning with augmentation (rotation, etc.)

**Dataset**: Freshwater Fish Disease Aquaculture (South Asia)
- Classes: Bacterial Red disease, Fungal infection, Parasitic disease, Healthy
- Splits: 90% train, 10% validation

**Notebook Workflow**:
1. Load and split fish disease images
2. Build ResNet50 transfer learning model
3. Train with data augmentation
4. Evaluate on test set (accuracy, F1, precision, recall)
5. Generate confusion matrix and training curves
6. Run inference on new images

**Usage**:
Open and run `src/health_monitoring/fresh-water-fish-disease-classification.ipynb` in Jupyter

### 4. Biomass Estimation (`src/biomass_estimation/`)

**Purpose**: Predict fish weight and biomass from images or measurements.

**Notebook**: `Train.ipynb` - Complete training and evaluation pipeline

**Usage**:
Open and run `src/biomass_estimation/Train.ipynb` in Jupyter

### 5. Water Quality Forecasting (`src/water_quality_prediction/`)

**Purpose**: Forecast water quality parameters (temperature, pH, dissolved oxygen, etc.) for aquaculture systems.

**Approach**: Time series forecasting with deep learning (LSTM/GRU)

**Dataset**: Aquaponics and fish pond datasets

**Notebook**: `aquaponics-fish-pond-datasets-forecasting-model.ipynb`

**Usage**:
Open and run `src/water_quality_prediction/aquaponics-fish-pond-datasets-forecasting-model.ipynb` in Jupyter

## 🔧 Configuration

### Smart Feeding Configuration (`src/smart_feeding/config.py`)

```python
CONFIG = {
    "exp_name": "Cnn6_Colab_Run",           # Experiment name
    "model_type": "Cnn6",                   # Model architecture
    "batch_size": 128,                      # Batch size
    "num_epochs": 100,                      # Training epochs
    "learning_rate": 1e-3,                  # Initial learning rate
    "sample_rate": 32000,                   # Audio sample rate (Hz)
    "classes_num": 4,                       # Number of classes
    "train_ratio": 0.8,                     # Train/test split
    "seed": 42,                             # Random seed
    "num_workers": 4,                       # DataLoader workers
    "workspace_dir": "./Fish_workspace",    # Output directory
}
```

### YOLO Configuration (`config/yolov8_config.yaml`)

```yaml
path: /path/to/fish-skeleton-yolo

train: train/images
val: valid/images
test: test/images

nc: 2                    # Number of classes
names:
  0: MrTri-descendants
  1: fish

kpt_shape: [7, 3]       # 7 keypoints, 3 values (x, y, visibility)

skeleton:
  - [1, 2]              # Skeleton connections for visualization
  - [2, 3]
  - [1, 4]
  - [3, 1]
  - [4, 5]
  - [5, 6]
  - [6, 7]
```

## 📊 Training & Evaluation

### Smart Feeding Training

```bash
cd src/smart_feeding

# Train with default config
python train.py

# Train with custom config (modify config.py first)
python train.py

# Evaluate and generate visualizations
python visualize_results.py --ckpt <path_to_checkpoint> --batch-size 32
```

**Output**:
- Checkpoints: `results/smart_feeding/<exp_name>/save_models/`
- Training history: `results/smart_feeding/<exp_name>/training_history.csv`
- Plots: `results/smart_feeding/<exp_name>/plots/`

### YOLO Tracking Training

```bash
# Open Jupyter and run Tracking_counting.ipynb
jupyter notebook src/tracking_counting/Tracking_counting.ipynb
```

**Key cells**:
1. Install ultralytics + dependencies
2. Load pre-converted YOLO dataset
3. Train YOLOv8n-pose
4. Visualize predictions
5. Generate video
6. Evaluate metrics

## 📈 Metrics

### Smart Feeding
- **mAP (mean Average Precision)**: Per-class and overall
- **AUC (Area Under Curve)**: Per-class ROC-AUC
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class prediction breakdown

### Fish Tracking
- **mAP50**: mAP at IoU=0.50 (boxes and keypoints)
- **mAP50-95**: mAP averaged over IoU=[0.50:0.95]
- **Per-class metrics**: For each fish class

### Disease Classification
- **Accuracy**: Overall correctness
- **F1 Score**: Weighted average precision-recall
- **Precision & Recall**: Per-disease metrics
- **Confusion Matrix**: Misclassification patterns

## 🔍 Troubleshooting

### Import Errors

If you encounter import errors:
```bash
# Reinstall core dependencies
pip install --upgrade torch torchvision torchaudio
pip install --upgrade tensorflow
pip install --upgrade ultralytics
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU-only mode (modify code)
device = 'cpu'  # Instead of 'cuda' if available
```

### Dataset Not Found

Ensure all data directories exist:
```bash
# Check structure
ls -la data/Fish_feeding_sounds/
ls -la data/Couting_tracking_biomass/fish-skeleton-yolo/
```

### Model Training Slow

- Reduce `batch_size` if out of memory
- Set `num_workers=0` if DataLoader hangs
- Use smaller model variant (e.g., yolov8n instead of yolov8x)

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📧 Contact & Support

For questions, issues, or collaboration:
- **GitHub Issues**: Report bugs and feature requests
- **Email**: Contact repository maintainers
- **Documentation**: See module-specific READMEs in each `src/` subdirectory

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- ResNet50 and transfer learning methodology
- Librosa for audio processing
- PyTorch and TensorFlow communities
- Public aquaculture datasets and contributors

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Audio Tutorial](https://pytorch.org/audio/stable/index.html)
- [Librosa Documentation](https://librosa.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## 🔗 Related Resources

- [Public Datasets](data/public_datasets.md)
- [Configuration Guide](config/)
- [Training Results](results/)

---

**Last Updated**: November 2025
**Repository**: [AquaSense GitHub](https://github.com/hotrungtruc/AquaSense-Intelligent-Aquaculture-ML-Platform)
**Status**: Active Development

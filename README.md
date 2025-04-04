# QUACLRS: QUasi-supervised Audio Classification by Learning Representations from Spectrograms

This project implements multiple CNN architectures for audio spectrogram classification using the extended UrbanSound8K dataset. The implementation includes comprehensive training, evaluation, and visualization tools.

## Project Structure

```
urbansound_extended/
├── spectrograms/               # Directory containing spectrogram images
│   ├── fold1/                  # Fold 1 spectrograms
│   ├── fold2/                  # Fold 2 spectrograms
│   └── ...                     # ... up to fold10
├── spectrograms_balanced_no_sirens.csv  # Dataset metadata
├── utils/                      # Utility modules
│   ├── __init__.py             # Package initialization
│   ├── dataset.py              # Dataset and dataloader utilities
│   ├── models.py               # Model architecture definitions
│   ├── training.py             # Training and evaluation utilities
│   └── visualization.py        # Visualization utilities
├── train_kfold.py              # K-fold cross-validation training script
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Dataset

The extended UrbanSound8K dataset contains 57,918 spectrogram images with the following characteristics:
- **Image dimensions**: 1000×400 pixels, RGBA format
- **13 Classes**: dog_bark, children_playing, car_horn, air_conditioner, street_music, gun_shot, engine_idling, jackhammer, drilling, ambulance, firetruck, police, traffic
- **Folds**: 10 folds for cross-validation
- **Augmentations**: Various augmentations already applied (timestretch, pitchshift, specaugment, patchaugment, noise)

## Supported Models

This project supports training the following CNN architectures:
- AlexNet
- VGG16
- ResNet18
- MobileNetV2
- InceptionV3
- EfficientNet-B0
- ConvNeXt-Tiny

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a model with k-fold cross-validation

```bash
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30
```

### Training on a specific fold

```bash
python train_kfold.py --model_name mobilenet --pretrained --fold 1
```

### Training with mixed precision

```bash
python train_kfold.py --model_name efficientnet --pretrained --use_amp
```

### Full list of available options

```
usage: train_kfold.py [-h] [--csv_file CSV_FILE] [--spectrograms_dir SPECTROGRAMS_DIR]
                    --model_name {alexnet,vgg16,resnet18,mobilenet,inception,efficientnet,convnext}
                    [--pretrained] [--freeze_features] [--batch_size BATCH_SIZE]
                    [--num_workers NUM_WORKERS] [--learning_rate LEARNING_RATE]
                    [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS] [--patience PATIENCE]
                    [--scheduler {plateau,cosine}] [--device {cuda,cpu}] [--use_amp]
                    [--distributed] [--output_dir OUTPUT_DIR] [--fold FOLD]

Train audio spectrogram classification models with K-fold cross-validation

optional arguments:
  -h, --help            show this help message and exit
  --csv_file CSV_FILE   Path to the CSV file with annotations
  --spectrograms_dir SPECTROGRAMS_DIR
                        Directory containing spectrogram images
  --model_name {alexnet,vgg16,resnet18,mobilenet,inception,efficientnet,convnext}
                        Model architecture to use
  --pretrained          Use pretrained weights for the model
  --freeze_features     Freeze feature extraction layers
  --batch_size BATCH_SIZE
                        Batch size for training
  --num_workers NUM_WORKERS
                        Number of worker threads for data loading
  --learning_rate LEARNING_RATE
                        Initial learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay for optimizer
  --epochs EPOCHS       Number of epochs to train for
  --patience PATIENCE   Patience for early stopping
  --scheduler {plateau,cosine}
                        Learning rate scheduler to use
  --device {cuda,cpu}   Device to use for training (cuda/cpu)
  --use_amp             Use automatic mixed precision training
  --output_dir OUTPUT_DIR
                        Directory to save outputs
  --fold FOLD           Specific fold to train on (None for all folds)
```

## Training Output

For each trained model, the following outputs are generated:
- **Model checkpoints**: Regular and best model checkpoints
- **Learning curves**: Loss and accuracy plots
- **Confusion matrix**: Visualization of classification performance
- **ROC curves**: ROC and AUC metrics for each class
- **Classification report**: Precision, recall, and F1-score for each class
- **Grad-CAM visualizations**: Attention maps for sample images

## Hardware Requirements

The code is optimized for use with NVIDIA GPUs. Recommended hardware:
- 4 NVIDIA RTX 2080 Ti GPUs
- 16GB RAM
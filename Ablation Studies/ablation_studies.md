# Ablation Studies for Audio Spectrogram Classification

This document outlines various training configurations for conducting ablation studies on the extended UrbanSound dataset. Each configuration tests different aspects of the model and training process to determine their impact on performance.

## Model Architecture Variations ✅

Test how different CNN architectures perform on the spectrogram classification task.

```bash
# AlexNet - Lighter architecture, fewer parameters
python train_kfold.py --model_name alexnet --pretrained --batch_size 32 --epochs 30 ✅

# VGG16 - Deeper network with more parameters
python train_kfold.py --model_name vgg16 --pretrained --batch_size 32 --epochs 30 ✅

# ResNet18 - Residual connections, moderate parameter count
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 ✅

# MobileNet - Lightweight, efficient architecture
python train_kfold.py --model_name mobilenet --pretrained --batch_size 32 --epochs 30 ✅

# InceptionV3 - Parallel convolutions at different scales
python train_kfold.py --model_name inception --pretrained --batch_size 32 --epochs 30 ✅

# EfficientNet - Optimized scaling of network width/depth/resolution
python train_kfold.py --model_name efficientnet --pretrained --batch_size 32 --epochs 30 ✅

# ConvNeXt - Modern architecture with depthwise convolutions 
python train_kfold.py --model_name convnext --pretrained --batch_size 32 --epochs 30 ✅
```

## Transfer Learning Impact ✅

Test the effect of using pretrained weights vs. training from scratch.

```bash
# With pretrained weights (transfer learning)
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 ✅

# Without pretrained weights (training from scratch)
python train_kfold.py --model_name resnet18 --batch_size 32 --epochs 30 ✅
```

## Feature Extraction vs. Fine-tuning ✅

Test whether freezing feature extraction layers improves performance.

```bash
# Full fine-tuning (all layers trainable)
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 ✅

# Feature extraction (only classifier layers are trainable)
python train_kfold.py --model_name resnet18 --pretrained --freeze_features --batch_size 32 --epochs 30 ✅
```

## Learning Rate Variations ✅
 
Test how different learning rates affect convergence and final performance.

```bash
# Lower learning rate
python train_kfold.py --model_name resnet18 --pretrained --learning_rate 0.0001 --batch_size 32 --epochs 30 ✅

# Default learning rate
python train_kfold.py --model_name resnet18 --pretrained --learning_rate 0.001 --batch_size 32 --epochs 30 ✅

# Higher learning rate
python train_kfold.py --model_name resnet18 --pretrained --learning_rate 0.01 --batch_size 32 --epochs 30 N/A
```

## Batch Size Impact ✅

Test how different batch sizes affect training dynamics and generalization.

```bash
# Small batch size
python train_kfold.py --model_name resnet18 --pretrained --batch_size 8 --epochs 30 N/A

# Medium batch size
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 N/A

# Large batch size
python train_kfold.py --model_name resnet18 --pretrained --batch_size 64 --epochs 30 ✅

# Very large batch size (if VRAM allows)
python train_kfold.py --model_name resnet18 --pretrained --batch_size 128 --epochs 30 N/A
```

## Learning Rate Scheduler Comparison ✅

Test different learning rate scheduling strategies.

```bash
# ReduceLROnPlateau scheduler (adaptive)
python train_kfold.py --model_name resnet18 --pretrained --scheduler plateau --batch_size 32 --epochs 30 ✅ default

# CosineAnnealing scheduler (gradual decay)
python train_kfold.py --model_name resnet18 --pretrained --scheduler cosine --batch_size 32 --epochs 30 ✅
```

## Weight Decay (Regularization) Impact 

Test different regularization strengths through weight decay.

```bash
# Very low weight decay
python train_kfold.py --model_name resnet18 --pretrained --weight_decay 1e-6 --batch_size 32 --epochs 30 TBD

# Default weight decay
python train_kfold.py --model_name resnet18 --pretrained --weight_decay 1e-5 --batch_size 32 --epochs 30 ✅

# Higher weight decay
python train_kfold.py --model_name resnet18 --pretrained --weight_decay 1e-4 --batch_size 32 --epochs 30 TBD
```

## Early Stopping Patience

Test different patience settings for early stopping.

```bash
# Quick early stopping (less patience)
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --patience 3 N/A

# Default patience
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --patience 7 ✅

# More patience
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --patience 12 TBD
```

## Mixed Precision Training ✅

Test if mixed precision training affects performance while speeding up training.

```bash
# Without mixed precision 
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 ✅

# With mixed precision
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --use_amp ✅
```

## Single Fold vs. Full Cross-Validation

Test training on specific folds vs. all folds to evaluate consistency.

```bash
# Train on all folds (full 10-fold cross-validation)
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 ✅

# Train only on fold 1
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --fold 1 TBD

# Train only on fold 5
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --fold 5 TBD
```

## Device Testing ✅

Test training on different hardware.

```bash
# Train on GPU (default)
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --device cuda ✅

# Train on CPU
python train_kfold.py --model_name resnet18 --pretrained --batch_size 32 --epochs 30 --device cpu N/A
```

## Experiment Tracking

For each experiment, track the following metrics:
1. Test accuracy (average across folds)
2. Per-class precision, recall, and F1-score
3. Training time
4. Model size
5. Best epoch (when early stopping occurred)

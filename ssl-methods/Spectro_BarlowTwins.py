#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
from collections import deque
import math

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# LARS Optimizer implementation (Layer-wise Adaptive Rate Scaling)
class LARS(optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9, weight_decay=0.0001, trust_coefficient=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, 
                        trust_coefficient=trust_coefficient, eps=eps)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get params
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                # Get parameters
                momentum_buffer = state['momentum_buffer']
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                lr = group['lr']
                trust_coefficient = group['trust_coefficient']
                eps = group['eps']
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # LARS trust ratio
                if p.ndim > 1:  # Skip LARS for biases, batch norms, etc.
                    w_norm = torch.norm(p.data)
                    g_norm = torch.norm(grad.data)
                    
                    if w_norm > 0 and g_norm > 0:
                        trust_ratio = trust_coefficient * w_norm / (g_norm + weight_decay * w_norm + eps)
                        # Apply LARS
                        grad = grad.mul(trust_ratio)
                    else:
                        # No LARS if norms are 0
                        trust_ratio = 1.0
                else:
                    trust_ratio = 1.0
                
                # Update with momentum
                momentum_buffer.mul_(momentum).add_(grad)
                
                # SGD update
                p.data.add_(momentum_buffer, alpha=-lr)
                
        return loss

# Dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, folds, csv_file, transform=None, use_augmentations=True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_augmentations = use_augmentations
        
        # Read the CSV file with the spectrogram information
        self.annotations = pd.read_csv(csv_file)
        
        # Handle single or multiple folds
        if isinstance(folds, int):
            folds = [folds]
            
        # Filter by folds
        self.file_list = self.annotations[self.annotations['fold'].isin(folds)]
        
        # Filter out augmentations if specified
        if not use_augmentations:
            self.file_list = self.file_list[self.file_list['augmentation'] == 'original']
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Get the file path
        img_name = self.file_list.iloc[idx]['spec_file_name']
        fold = self.file_list.iloc[idx]['fold']
        label = self.file_list.iloc[idx]['classID']
        
        # Construct the full path to the image
        img_path = os.path.join(self.root_dir, f'fold{fold}', img_name)
        
        # Load the image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            if self.use_augmentations:
                # For Barlow Twins, we need two differently augmented views of the same image
                z1 = self.transform(image)
                z2 = self.transform(image)
                return z1, z2, label
            else:
                # Always apply transform for validation
                image = self.transform(image)
        
        return image, label

# Barlow Twins Projection Head 
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=8192, output_dim=8192):
        super().__init__()
        # Barlow Twins uses a 3-layer MLP with larger dimensionality
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # Last BN layer has no affine transform
        )
    
    def forward(self, x):
        return self.layers(x)

# Barlow Twins model
class BarlowTwins(nn.Module):
    def __init__(self, backbone=None, hidden_dim=8192, output_dim=8192, lambd=0.005):
        super().__init__()
        
        # Create backbone if not provided
        if backbone is None:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Identity()  # Remove classification layer
        
        self.backbone = backbone
        self.projection_head = ProjectionHead(
            input_dim=2048,  # ResNet50 output dimension
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Barlow Twins lambda parameter for the off-diagonal elements
        self.lambd = lambd
        
    def forward(self, x1, x2=None, get_features=False):
        # Extract features only
        if get_features:
            return self.backbone(x1)
            
        # Forward pass through the backbone for both views
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        
        # Forward pass through projection head
        z1 = self.projection_head(f1)
        z2 = self.projection_head(f2)
        
        return z1, z2, f1
    
    @torch.no_grad()
    def get_features(self, x):
        """Get features from the backbone for downstream tasks"""
        return self.backbone(x)
    
    def barlow_twins_loss(self, z1, z2):
        """
        Implements the Barlow Twins loss function
        
        Args:
            z1, z2: Batch of projected features from the two views
            
        Returns:
            loss: The Barlow Twins loss
        """
        # Normalize along the batch dimension
        # Important: use proper normalization to prevent huge loss values
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

        N = z1.size(0)  # Batch size
        D = z1.size(1)  # Representation dimension
        
        # Cross-correlation matrix - properly normalized by batch size
        c = torch.mm(z1_norm.T, z2_norm) / N
        
        # Identity matrix as target
        eye = torch.eye(D, device=z1.device)
        
        # Loss: MSE between cross-correlation and identity matrix
        # On-diagonal terms (should be close to 1)
        on_diag = ((c - eye).diagonal().pow(2)).sum()
        
        # Off-diagonal terms (should be close to 0)
        off_diag = ((c - eye).pow(2)).sum() - on_diag
        
        # Combine with lambda weighting for off-diagonal terms
        # Scale the loss to have reasonable magnitude
        loss = on_diag + self.lambd * off_diag
        
        return loss

# Classifier for downstream task
class Classifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=14):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Training function
def train(fold, class_names, num_classes, debug=False, batch_size=32, num_epochs=30):
    """
    Train the Barlow Twins model on a single fold
    
    Args:
        fold: Fold number to use as validation
        class_names: List of class names
        num_classes: Number of classes
        debug: Enable debugging mode
        batch_size: Batch size
        num_epochs: Number of epochs
    """
    # Create timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_barlowtwins_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable anomaly detection if in debug mode
    if debug:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled for debugging")
    
    # Define transformations for training - using the same strong augmentations from SimCLR
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for validation (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data loaders
    train_folds = [f for f in range(1, 11) if f != fold]
    
    train_ds = SpectrogramDataset(
        './spectrograms', 
        train_folds, 
        './spectrograms_balanced.csv', 
        transform, 
        use_augmentations=True
    )
    
    val_ds = SpectrogramDataset(
        './spectrograms', 
        [fold], 
        './spectrograms_balanced.csv', 
        eval_transform, 
        use_augmentations=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Initialize models
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    
    # Reduced output dimension to improve training stability
    barlow_twins = BarlowTwins(backbone, hidden_dim=4096, output_dim=4096, lambd=0.0051).to(device)
    classifier = Classifier(input_dim=2048, num_classes=num_classes).to(device)
    
    # Learning rate setup based on original paper
    base_lr = 0.2  # Base LR adjusted for batch size
    
    # Use different learning rates for backbone and projection head
    # Exclude batch norm parameters from weight decay
    params = []
    # Backbone params
    for name, param in barlow_twins.backbone.named_parameters():
        if 'bn' in name:
            # No weight decay for BN parameters
            params.append({'params': param, 'weight_decay': 0, 'name': f'backbone.{name}'})
        else:
            params.append({'params': param, 'weight_decay': 1e-6, 'name': f'backbone.{name}'})
    
    # Projection head
    for name, param in barlow_twins.projection_head.named_parameters():
        if 'bn' in name:
            params.append({'params': param, 'weight_decay': 0, 'name': f'projection.{name}'})
        else:
            params.append({'params': param, 'weight_decay': 1e-6, 'name': f'projection.{name}'})
    
    # Classifier
    for name, param in classifier.named_parameters():
        params.append({'params': param, 'weight_decay': 1e-6, 'name': f'classifier.{name}'})
    
    # Create LARS optimizer with correct scaling for batch size
    scaled_lr = base_lr * (batch_size / 256)
    optimizer = LARS(
        params,
        lr=scaled_lr,
        momentum=0.9,
        weight_decay=1e-6,
        trust_coefficient=0.001
    )
    
    # Cosine learning rate scheduler with warm-up
    warmup_epochs = 10
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_schedule(step):
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # Classification loss
    classification_criterion = torch.nn.CrossEntropyLoss()
    
    # Metrics storage
    train_losses, val_losses = [], []
    val_accuracies, all_preds, all_labels = [], [], []
    
    for epoch in range(num_epochs):
        # TRAINING
        barlow_twins.train()
        classifier.train()
        epoch_train_loss = 0.0
        bt_losses = []
        cls_losses = []
        
        for batch_idx, (z1, z2, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            z1, z2, labels = z1.to(device), z2.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass for Barlow Twins
            p1, p2, features = barlow_twins(z1, z2)
            
            # Barlow Twins loss
            bt_loss = barlow_twins.barlow_twins_loss(p1, p2)
            bt_losses.append(bt_loss.item())
            
            # Classification loss - using features from first view
            with torch.no_grad():
                features = barlow_twins.get_features(z1)
            
            # Make predictions using these features
            logits_cls = classifier(features.detach())
            cls_loss = classification_criterion(logits_cls, labels)
            cls_losses.append(cls_loss.item())
            
            # Total loss
            loss = bt_loss + cls_loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Track metrics
            epoch_train_loss += loss.item() * z1.size(0)
            
            # Report training progress
            if batch_idx % 10 == 0:
                # Calculate average losses to show on progress bar
                avg_bt_loss = sum(bt_losses[-10:]) / min(10, len(bt_losses))
                avg_cls_loss = sum(cls_losses[-10:]) / min(10, len(cls_losses))
                
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"BT Loss: {avg_bt_loss:.4f}, "
                      f"Cls Loss: {avg_cls_loss:.4f}, "
                      f"Total Loss: {avg_bt_loss + avg_cls_loss:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Calculate average training loss
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # VALIDATION
        barlow_twins.eval()
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        epoch_preds = []
        epoch_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Extract features
                features = barlow_twins.get_features(images)
                
                # Make predictions
                logits = classifier(features)
                loss = classification_criterion(logits, labels)
                
                # Track metrics
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, predictions = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
                # Store predictions and labels for confusion matrix
                epoch_preds.extend(predictions.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        all_preds.append(epoch_preds)
        all_labels.append(epoch_labels)
        
        # Report epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'barlow_twins': barlow_twins.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, f"{results_dir}/checkpoint_{epoch}.pth")
    
    # Save the final model
    torch.save({
        'barlow_twins': barlow_twins.state_dict(),
        'classifier': classifier.state_dict(),
        'val_accuracy': val_accuracies[-1],
        'class_names': class_names
    }, f"{results_dir}/fold_{fold}_model.pth")
    
    # Save loss and accuracy plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss Curve for Fold {fold}')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(f'Accuracy Curve for Fold {fold}')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/fold_{fold}_plots.png")
    
    # Generate confusion matrix for the final epoch
    final_cm = confusion_matrix(all_labels[-1], all_preds[-1])
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fold {fold}')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/fold_{fold}_confusion_matrix.png")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels[-1], all_preds[-1], target_names=class_names))
    
    return val_accuracies[-1]

def main():
    """
    Main function to start the training
    """
    parser = argparse.ArgumentParser(description='Train Barlow Twins on spectrograms')
    parser.add_argument('--fold', type=int, help='Fold to use as validation (1-10)', required=False)
    parser.add_argument('--debug', action='store_true', help='Enable anomaly detection for debugging')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    args = parser.parse_args()
    
    # Enable anomaly detection if requested
    if args.debug:
        print("Enabling anomaly detection for debugging")
        torch.autograd.set_detect_anomaly(True)
    
    # Set random seed
    set_seed(42)
    
    # Get class names and auto-detect num_classes
    print("Loading data from ./spectrograms_balanced.csv")
    full_annotations = pd.read_csv('./spectrograms_balanced.csv')
    class_names = sorted(full_annotations['class'].unique())
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")
    
    num_folds = 10
    
    # Train specific fold or all folds
    if args.fold is not None:
        # Train only the specified fold
        fold = args.fold
        print(f"\n=== Training Fold {fold}/{num_folds} ===")
        train(fold, class_names, num_classes, args.debug, args.batch_size, args.epochs)
    else:
        # Train all folds one by one
        all_fold_accuracies = []
        
        for fold in range(1, num_folds+1):
            print(f"\n=== Training Fold {fold}/{num_folds} ===")
            accuracy = train(fold, class_names, num_classes, args.debug, args.batch_size, args.epochs)
            all_fold_accuracies.append(accuracy)
        
        # Report average accuracy
        avg_accuracy = np.mean(all_fold_accuracies)
        print(f"\nAverage accuracy across {num_folds} folds: {avg_accuracy:.2f}%")

# Inference class for making predictions using a trained model
class SpectrogramClassifier:
    def __init__(self, checkpoint_path, class_names, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model components
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        
        self.barlow_twins = BarlowTwins(backbone).to(self.device)
        self.classifier = Classifier(input_dim=2048, num_classes=len(class_names)).to(self.device)
        self.class_names = class_names
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.barlow_twins.load_state_dict(checkpoint['barlow_twins'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        
        # Set to evaluation mode
        self.barlow_twins.eval()
        self.classifier.eval()
        
        # Define transforms (similar to validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, img_path, verbose=True):
        """
        Predict class for a spectrogram image
        Args:
            img_path: Path to the spectrogram image
            verbose: Whether to print detailed information
        Returns:
            Class name, probability, and all class probabilities
        """
        # Load and transform the image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            features = self.barlow_twins.get_features(img_tensor)
            logits = self.classifier(features)
            probabilities = F.softmax(logits, dim=1)[0]
        
        # Get predicted class and probability
        pred_idx = torch.argmax(probabilities).item()
        pred_class = self.class_names[pred_idx]
        confidence = probabilities[pred_idx].item() * 100
        
        if verbose:
            print(f"Prediction: {pred_class}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Print top-3 predictions
            top3_values, top3_indices = torch.topk(probabilities, 3)
            print("\nTop 3 predictions:")
            for i in range(3):
                idx = top3_indices[i].item()
                prob = top3_values[i].item() * 100
                print(f"{self.class_names[idx]}: {prob:.2f}%")
        
        return pred_class, confidence, probabilities.cpu().numpy()

# Visualize prediction
def visualize_prediction(probs, img_path, class_names):
    """
    Visualize prediction with the spectrogram image
    Args:
        probs: Numpy array of class probabilities
        img_path: Path to the spectrogram image
        class_names: List of class names
    """
    # Load image
    img = Image.open(img_path).convert('RGB')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot image
    ax1.imshow(img)
    ax1.set_title('Spectrogram')
    ax1.axis('off')
    
    # Plot probabilities
    sorted_idx = np.argsort(probs)[::-1]
    top_classes = [class_names[i] for i in sorted_idx[:5]]
    top_probs = [probs[i] * 100 for i in sorted_idx[:5]]
    
    ax2.barh(top_classes, top_probs, color='skyblue')
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top Predictions')
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()

# Function to evaluate model on a specific fold
def evaluate_fold(model_path, fold, class_names):
    """
    Evaluate a trained model on a specific fold
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the classifier
    classifier = SpectrogramClassifier(model_path, class_names, device)
    
    # Create the validation dataset
    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_ds = SpectrogramDataset(
        './spectrograms', 
        [fold], 
        './spectrograms_balanced.csv', 
        eval_transform, 
        use_augmentations=False
    )
    
    # Calculate metrics
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    # Process each image
    for i in tqdm(range(len(val_ds))):
        img, label = val_ds[i]
        img_tensor = img.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            features = classifier.barlow_twins.get_features(img_tensor)
            logits = classifier.classifier(features)
            _, predicted = torch.max(logits, 1)
            
        pred_idx = predicted.item()
        all_preds.append(pred_idx)
        all_labels.append(label)
        
        # Update accuracy
        if pred_idx == label:
            correct += 1
        total += 1
    
    # Calculate metrics
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nResults for fold {fold}:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fold {fold}')
    plt.show()
    
    return accuracy, f1

# Function to make a prediction for a single file
def predict(img_path, model_path, class_names):
    """
    Make a prediction for a single image
    """
    # Create classifier
    classifier = SpectrogramClassifier(model_path, class_names)
    
    # Make prediction
    _, _, probs = classifier.predict(img_path)
    
    # Visualize
    visualize_prediction(probs, img_path, class_names)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

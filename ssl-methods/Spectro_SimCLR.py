#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1
num_folds = 10
batch_size = 32
feature_dim = 2048

# Relative paths - can be adjusted as needed
root_dir = './spectrograms/'
balanced_csv_file = "./spectrograms_balanced.csv"

# Get class names and auto-detect num_classes
print(f"Loading data from {balanced_csv_file}")
full_annotations = pd.read_csv(balanced_csv_file)
class_names = sorted(full_annotations['class'].unique())
num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")
print(f"Using device: {device}")

# Modified Dataset Class for spectrogram data
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
            
        print(f"Loaded {len(self.file_list)} spectrogram files")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        row = self.file_list.iloc[idx]
        
        # Construct the path to the spectrogram image
        img_path = os.path.join(self.root_dir, f'fold{row["fold"]}', row['spec_file_name'])
        
        # Load the image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image can't be loaded
            image = Image.new('RGB', (224, 224))
            
        # Get the class label
        label = row['classID']
        
        # Apply transformations
        if self.transform:
            if self.use_augmentations:
                # For contrastive learning, create two views of the same image
                xi = self.transform(image)
                xj = self.transform(image)
                return xi, xj, label
            else:
                # Always apply transform for validation
                image = self.transform(image)
        
        return image, label

# Model Components
class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead()
    def forward(self, x):
        features = self.backbone(x)
        return self.projection(features)

class Classifier(torch.nn.Module):
    def __init__(self, input_dim=2048, num_classes=10):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# Loss Function
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.T) / self.temperature
        
        # Create labels: positives are the N off-diagonal elements
        labels = torch.cat([
            torch.arange(N, 2*N, device=z.device),
            torch.arange(0, N, device=z.device)
        ])
        
        # Mask out self-similarity
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)  # Use large negative value instead of inf
        
        # Compute loss
        loss = self.criterion(sim, labels)
        return loss

def train():
    # Initialize backbone (keep original ImageNet weights)
    backbone = models.resnet50(pretrained=True)
    backbone.fc = torch.nn.Identity()
    
    # Data transformations for contrastive learning
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluation transformation (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Using the balanced CSV file
    selected_csv = balanced_csv_file
    print(f"Using CSV file: {selected_csv}")
    
    # Create timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Store results across folds
    all_fold_results = []

    # K-fold cross validation
    for fold in range(1, num_folds+1):
        print(f"\n=== Fold {fold}/{num_folds} ===")
        
        # Clear GPU cache to prevent memory buildup between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Data loaders
        train_ds = SpectrogramDataset(
            root_dir, 
            [f for f in range(1, 11) if f != fold], 
            selected_csv, 
            transform, 
            use_augmentations=True
        )
        
        val_ds = SpectrogramDataset(
            root_dir, 
            [fold], 
            selected_csv, 
            eval_transform, 
            use_augmentations=False
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Model components
        simclr = SimCLR(backbone).to(device)
        classifier = Classifier(input_dim=feature_dim, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(list(simclr.parameters()) + list(classifier.parameters()), lr=3e-4)
        contrastive_criterion = NTXentLoss()
        classification_criterion = torch.nn.CrossEntropyLoss()

        # Metrics storage
        train_losses, val_losses = [], []
        val_accuracies, all_preds, all_labels = [], [], []

        for epoch in range(num_epochs):
            # TRAINING
            simclr.train()
            classifier.train()
            epoch_train_loss = 0.0
            
            for batch_idx, (xi, xj, labels) in enumerate(train_loader):
                xi, xj, labels = xi.to(device), xj.to(device), labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass for contrastive learning
                zi = simclr(xi)
                zj = simclr(xj)
                contrastive_loss = contrastive_criterion(zi, zj)
                
                # Extracting features for classification
                with torch.no_grad():
                    features_i = simclr.backbone(xi)
                    features_j = simclr.backbone(xj)
                
                # Classification forward pass
                logits_i = classifier(features_i)
                logits_j = classifier(features_j)
                
                # Classification loss (average of both views)
                cls_loss_i = classification_criterion(logits_i, labels)
                cls_loss_j = classification_criterion(logits_j, labels)
                cls_loss = (cls_loss_i + cls_loss_j) / 2
                
                # Combined loss (weighted sum)
                loss = contrastive_loss * 0.7 + cls_loss * 0.3
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_train_loss += loss.item()
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
            # VALIDATION
            simclr.eval()
            classifier.eval()
            epoch_val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Extract features from a single view (validation only has one view)
                    features = simclr.backbone(inputs)
                    
                    # Use the features for evaluation
                    logits = classifier(features)
                    
                    # Compute loss
                    val_loss = classification_criterion(logits, labels)
                    epoch_val_loss += val_loss.item()
                    
                    # Compute accuracy
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Store predictions and labels for confusion matrix
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Compute average losses and accuracy
            epoch_train_loss /= len(train_loader)
            epoch_val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(accuracy)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")
        
        # Save fold results
        fold_results = {
            "fold": fold,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "all_preds": all_preds,
            "all_labels": all_labels
        }
        all_fold_results.append(fold_results)
        
        # Visualize training metrics
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold} Losses')
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title(f'Fold {fold} Accuracy')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fold_{fold}_metrics.png")
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Fold {fold} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fold_{fold}_confusion.png")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Save model for this fold
        checkpoint_path = f"{results_dir}/fold_{fold}_model.pth"
        torch.save({
            'simclr': simclr.state_dict(),
            'classifier': classifier.state_dict(),
            'epoch': num_epochs,
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
        # Clean up to prevent CUDA memory leaks
        del simclr, classifier, optimizer, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Compute and report average performance across folds
    avg_accuracy = np.mean([fold_result["val_accuracies"][-1] for fold_result in all_fold_results])
    print(f"\nAverage accuracy across {num_folds} folds: {avg_accuracy:.2f}%")
    
    # Save summary plot of all folds
    plt.figure(figsize=(12, 6))
    for fold_result in all_fold_results:
        plt.plot(fold_result["val_accuracies"], label=f'Fold {fold_result["fold"]}')
    
    plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'Avg: {avg_accuracy:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy Across All Folds')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/all_folds_accuracy.png")
    print(f"Results saved to {results_dir}/")

# Inference class for making predictions using a trained model
class SpectrogramClassifier:
    def __init__(self, checkpoint_path, class_names, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model components
        backbone = models.resnet50(pretrained=False)
        backbone.fc = torch.nn.Identity()
        
        self.simclr = SimCLR(backbone).to(self.device)
        self.classifier = Classifier().to(self.device)
        self.class_names = class_names
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Debug: print keys
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        self.simclr.load_state_dict(checkpoint['simclr'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        
        # Set to evaluation mode
        self.simclr.eval()
        self.classifier.eval()
        
        # Define transforms (similar to validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Predict class for spectrogram image file
    def predict(self, img_path, verbose=True):
        """
        Predict class for a spectrogram image
        Args:
            img_path: Path to the spectrogram image
            verbose: Whether to print detailed information
        Returns:
            Class name, probability, and all class probabilities
        """
        # Load and preprocess image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, 0, []
            
        # Transform the image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            # Extract features
            features = self.simclr.backbone(img_tensor)
            
            # Get class logits
            logits = self.classifier(features)
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Get predicted class
            pred_idx = torch.argmax(probs).item()
            pred_class = self.class_names[pred_idx]
            pred_prob = probs[pred_idx].item()
            
            if verbose:
                print(f"Predicted class: {pred_class} with probability: {pred_prob:.4f}")
                
                # Print top 3 predictions
                top_k = 3
                top_probs, top_idxs = torch.topk(probs, top_k)
                print(f"\nTop {top_k} predictions:")
                for i in range(top_k):
                    idx = top_idxs[i].item()
                    cls = self.class_names[idx]
                    prob = top_probs[i].item()
                    print(f"{cls}: {prob:.4f}")
                    
            return pred_class, pred_prob, probs.cpu().numpy()

# Visualize prediction
def visualize_prediction(probs, img_path, class_names):
    """
    Visualize prediction with the spectrogram image
    Args:
        probs: Numpy array of class probabilities
        img_path: Path to the spectrogram image
        class_names: List of class names
    """
    img = Image.open(img_path).convert('RGB')
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    # Plot spectrogram
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title('Spectrogram')
    ax1.axis('off')
    
    # Plot probabilities
    ax2 = fig.add_subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run training with balanced dataset
    train()
    
    # Example of how to use the trained model for inference:
    """
    # Load a trained model
    model = SpectrogramClassifier('results_YYYYMMDD_HHMMSS/fold_1_model.pth', class_names)
    
    # Make a prediction
    img_path = './spectrograms/fold10/100648-1-orig.png'  # Adjust path as needed
    pred_class, pred_prob, all_probs = model.predict(img_path)
    
    # Visualize the prediction
    visualize_prediction(all_probs, img_path, class_names)
    """

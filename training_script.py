import os
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add argument parser for fold selection
parser = argparse.ArgumentParser(description='Train model on a specific fold')
parser.add_argument('--fold', type=int, default=10, help='Fold number to train on (1-10)')
#args = parser.parse_args()
args = parser.parse_args('') # Pass empty string in Jupyter

# Now you can access args.fold
print(f"Training on fold: {args.fold}")

# Device and Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 25
batch_size = 128
feature_dim = 512
hidden_dim = 512
projection_dim = 128
num_classes = 13
root_dir = './spectrograms/'
csv_file = "spectrograms_balanced_no_sirens.csv"

# Data validation
full_annotations = pd.read_csv(csv_file)
class_names = sorted(full_annotations['class'].unique())
print("Unique classes:", len(class_names))
print("Class ID range:", full_annotations['classID'].min(), full_annotations['classID'].max())
assert full_annotations['classID'].between(0, num_classes-1).all(), "Invalid class IDs detected"

# Dataset
class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, folds, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        if isinstance(folds, int):
            folds = [folds]
        self.file_list = self.annotations[self.annotations['fold'].isin(folds)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        row = self.file_list.iloc[idx]
        img_path = os.path.join(self.root_dir, f'fold{row["fold"]}', row['spec_file_name'])
        image = Image.open(img_path).convert('RGB')
        label = row['classID']
        if self.transform:
            xi = self.transform(image)
            xj = self.transform(image)
            return xi, xj, label
        return image, label

# Model components
class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim):
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
    def __init__(self, input_dim=feature_dim, num_classes=num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# Loss
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        labels = torch.cat([
            torch.arange(N, 2*N, device=z.device),
            torch.arange(0, N, device=z.device)
        ])
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)
        loss = self.criterion(sim, labels)
        return loss

# Training function with MobileNetV3
def train(fold):
    from torchvision.models import MobileNet_V3_Small_Weights
    
    # Validate fold number
    if fold < 1 or fold > 10:
        raise ValueError("Fold number must be between 1 and 10")
    
    print(f"\n{'='*40}")
    print(f"=== Training on Fold {fold} {'='*20}")
    print(f"{'='*40}\n")
    
    # 1. MobileNetV3-Small backbone
    backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    
    # 2. Modify classifier for feature extraction
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Linear(576, feature_dim),  # Original input features: 576
        torch.nn.Hardswish(inplace=True)
    )

    # 3. Input size for MobileNetV3
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])
    
    # Create training and validation datasets
    train_ds = UrbanSoundDataset(root_dir, [f for f in range(1,11) if f != fold], csv_file, transform)
    val_ds = UrbanSoundDataset(root_dir, [fold], csv_file, transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    
    simclr = SimCLR(backbone).to(device)
    classifier = Classifier().to(device)
    optimizer = torch.optim.Adam(
        list(simclr.parameters()) + list(classifier.parameters()), 
        lr=3e-4
    )
    criterion = NTXentLoss()

    train_losses, val_losses = [], []
    val_accuracies, all_preds, all_labels = [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        simclr.train()
        classifier.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (xi, xj, labels) in enumerate(train_loader):
            xi, xj, labels = xi.to(device), xj.to(device), labels.to(device)
            
            # Forward pass
            zi, zj = simclr(xi), simclr(xj)
            loss_contrastive = criterion(zi, zj)
            features = simclr.backbone(xi)
            logits = classifier(features)
            loss_classification = torch.nn.functional.cross_entropy(logits, labels)
            loss = loss_contrastive + 0.5 * loss_classification
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Progress reporting
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx + 1:03d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"CLoss: {loss_contrastive.item():.4f} | "
                      f"FLoss: {loss_classification.item():.4f} | "
                      f"LR: {current_lr:.2e}")

        # Epoch summary
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"\n  Training Summary | Epoch {epoch+1}")
        print(f"  Avg Loss: {avg_train_loss:.4f}")
        print(f"  Last Batch Loss: {loss.item():.4f}")

        # Validation phase
        simclr.eval()
        classifier.eval()
        val_loss, correct, total = 0, 0, 0
        print("\n  Validating...")
        with torch.no_grad():
            for batch_idx, (xi, _, labels) in enumerate(val_loader):
                xi, labels = xi.to(device), labels.to(device)
                features = simclr.backbone(xi)
                logits = classifier(features)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                # Metrics
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Validation progress
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(val_loader):
                    acc = 100 * (preds == labels).sum().item() / labels.size(0)
                    print(f"    Val Batch {batch_idx + 1:03d}/{len(val_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Batch Acc: {acc:.2f}%")

        # Validation summary
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        print(f"\n  Validation Summary | Epoch {epoch+1}")
        print(f"  Avg Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        print(f"  Current Best Acc: {max(val_accuracies):.2f}%")

    # Fold completion
    print(f"\n{'='*40}")
    print(f"=== Fold {fold} Completed ===")
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Loss Curves - Fold {fold}')
    plt.legend()
    plt.subplot(122)
    plt.plot(val_accuracies)
    plt.title(f'Validation Accuracy - Fold {fold}')
    plt.tight_layout()
    plt.savefig(f'fold_{fold}_metrics.png')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(f'fold_{fold}_confusion.png')
    plt.close()

    # Classification report
    print(f"\nClassification Report - Fold {fold}:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Model checkpoint
    torch.save({
        'simclr': simclr.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f'fold_{fold}_checkpoint.pth')

if __name__ == "__main__":
    train(args.fold)
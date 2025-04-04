import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class SpectrogramDataset(Dataset):
    """
    Dataset class for loading spectrogram images for audio classification.
    """
    def __init__(self, csv_file, base_dir, transform=None, fold=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            base_dir (str): Directory where spectrogram images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            fold (int or list, optional): If provided, only samples from this fold/folds will be included.
        """
        self.data_frame = pd.read_csv(csv_file)
        
        # Filter by fold if specified
        if fold is not None:
            if isinstance(fold, list):
                self.data_frame = self.data_frame[self.data_frame['fold'].isin(fold)]
            else:
                self.data_frame = self.data_frame[self.data_frame['fold'] == fold]
                
        self.base_dir = base_dir
        self.transform = transform
        
        # Create class to index mapping
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(sorted(self.data_frame['class'].unique()))
        }
        
        # Store the list of classes
        self.classes = sorted(self.data_frame['class'].unique())
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get row at the specified index
        row = self.data_frame.iloc[idx]
        
        # Get image path and class
        spec_file = row['spec_file_name']
        class_name = row['class']
        fold_num = row['fold']
        
        # Construct full image path
        img_path = os.path.join(self.base_dir, f'fold{fold_num}', spec_file)
        
        # Load image
        image = Image.open(img_path).convert('RGB')  # Convert RGBA to RGB
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            
        # Get class index
        label = self.class_to_idx[class_name]
        
        return image, label, spec_file
    
    def get_class_weights(self):
        """Compute class weights for imbalanced datasets"""
        class_counts = self.data_frame['class'].value_counts().to_dict()
        total = sum(class_counts.values())
        
        weights = {}
        for class_name, count in class_counts.items():
            weights[self.class_to_idx[class_name]] = total / (len(class_counts) * count)
            
        return weights


def create_dataloaders(csv_file, base_dir, fold_splits, batch_size=32, num_workers=4, model_name=None):
    """
    Create DataLoader objects for training, validation, and testing based on fold splits.
    
    Args:
        csv_file (str): Path to the CSV file with annotations.
        base_dir (str): Directory where spectrogram images are stored.
        fold_splits (dict): Dict with keys 'train', 'val', 'test' containing fold numbers.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for DataLoader.
        model_name (str, optional): Name of the model architecture to determine input size.
        
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoader objects.
    """
    # Define input size based on model
    if model_name == 'inception':
        # InceptionV3 requires larger input size
        input_size = (299, 299)
    else:
        # Standard size for other models
        input_size = (224, 224)
    
    # Define transformations for training and validation/testing
    train_transform = transforms.Compose([
        transforms.Resize(input_size),  # Resize to match model's required input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets for each split
    train_dataset = SpectrogramDataset(
        csv_file=csv_file,
        base_dir=base_dir,
        transform=train_transform,
        fold=fold_splits['train']
    )
    
    val_dataset = SpectrogramDataset(
        csv_file=csv_file,
        base_dir=base_dir,
        transform=val_transform,
        fold=fold_splits['val']
    )
    
    test_dataset = SpectrogramDataset(
        csv_file=csv_file,
        base_dir=base_dir,
        transform=val_transform,
        fold=fold_splits['test']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }

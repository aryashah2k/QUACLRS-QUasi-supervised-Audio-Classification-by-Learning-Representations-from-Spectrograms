import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import classification_report
import json
import time
from datetime import datetime

# Import utilities
from utils.dataset import create_dataloaders, SpectrogramDataset
from utils.training import train_one_epoch, validate, save_checkpoint
from utils.visualization import (
    plot_learning_curves, 
    plot_confusion_matrix, 
    plot_roc_curves, 
    plot_grad_cam_samples
)
from utils.models import create_model, get_target_layer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train audio spectrogram classification models with K-fold cross-validation')
    
    # Dataset parameters
    parser.add_argument('--csv_file', type=str, default='spectrograms_balanced_no_sirens.csv',
                        help='Path to the CSV file with annotations')
    parser.add_argument('--spectrograms_dir', type=str, default='spectrograms',
                        help='Directory containing spectrogram images')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['alexnet', 'vgg16', 'resnet18', 'mobilenet', 'inception', 'efficientnet', 'convnext'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for the model')
    parser.add_argument('--freeze_features', action='store_true',
                        help='Freeze feature extraction layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker threads for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=7,
                        help='Patience for early stopping')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler to use')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training across multiple GPUs')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train on (None for all folds)')
    
    return parser.parse_args()


def get_fold_splits(num_folds=10, fold=None):
    """
    Generate fold splits for cross-validation
    
    Args:
        num_folds: Total number of folds in the dataset
        fold: Specific fold to train on (None for all folds)
        
    Returns:
        fold_splits: List of dictionaries with train, val, and test fold assignments
    """
    all_folds = list(range(1, num_folds + 1))
    fold_splits = []
    
    if fold is not None:
        # For a specific fold, train on all other folds
        if fold < 1 or fold > num_folds:
            raise ValueError(f"Fold must be between 1 and {num_folds}")
        
        # Use fold for testing, fold-1 for validation (wrap around if needed)
        val_fold = fold - 1 if fold > 1 else num_folds
        test_fold = fold
        train_folds = [f for f in all_folds if f != val_fold and f != test_fold]
        
        fold_splits.append({
            'train': train_folds,
            'val': [val_fold],
            'test': [test_fold],
            'fold_num': fold
        })
    else:
        # For all folds, train on a leave-one-out basis
        for test_fold in all_folds:
            val_fold = test_fold - 1 if test_fold > 1 else num_folds
            train_folds = [f for f in all_folds if f != val_fold and f != test_fold]
            
            fold_splits.append({
                'train': train_folds,
                'val': [val_fold],
                'test': [test_fold],
                'fold_num': test_fold
            })
    
    return fold_splits


def train_model(args, fold_split):
    """
    Train a model on a specific fold split
    
    Args:
        args: Command-line arguments
        fold_split: Dictionary with train, val, and test fold assignments
        
    Returns:
        best_model_path: Path to the best model checkpoint
        metrics: Dictionary with evaluation metrics
    """
    # Set up output directory
    output_dir = os.path.join(args.output_dir, args.model_name, f'fold_{fold_split["fold_num"]}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        csv_file=args.csv_file,
        base_dir=args.spectrograms_dir,
        fold_splits=fold_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model_name
    )
    
    # Get number of classes
    num_classes = len(dataloaders['datasets']['train'].classes)
    class_names = dataloaders['datasets']['train'].classes
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_features=args.freeze_features
    )
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function with class weights to handle imbalance
    class_weights = dataloaders['datasets']['train'].get_class_weights()
    weights = torch.FloatTensor([class_weights[i] for i in range(num_classes)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Set up optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    else:  # cosine
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    
    # Set up automatic mixed precision if requested
    scaler = GradScaler() if args.use_amp else None
    
    # Initialize training variables
    best_accuracy = 0.0
    best_epoch = 0
    best_model_path = None
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping
    patience = args.patience
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=None if args.scheduler == 'plateau' else scheduler,
            scaler=scaler
        )
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion,
            device=device
        )
        
        # Update learning rate for ReduceLROnPlateau
        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        
        # Save training metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Save checkpoint and update best model
        current_model_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            accuracy=val_acc,
            loss=val_loss,
            fold=fold_split['fold_num'],
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        
        # Check if current model is the best
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            best_model_path = current_model_path
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Best Val Acc: {best_accuracy:.2f}% (Epoch {best_epoch+1})")
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot learning curves
    plot_learning_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        output_dir=args.output_dir,
        model_name=args.model_name,
        fold=fold_split['fold_num']
    )
    
    # Load best model for evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc, y_true, y_pred, y_probs = validate(
        model=model,
        dataloader=dataloaders['test'],
        criterion=criterion,
        device=device
    )
    
    # Generate plots and metrics
    cm_path = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        fold=fold_split['fold_num']
    )
    
    roc_path = plot_roc_curves(
        y_true=y_true,
        y_scores=y_probs,
        class_names=class_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        fold=fold_split['fold_num']
    )
    
    # Get classification report
    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generate Grad-CAM visualizations
    target_layer = get_target_layer(model, args.model_name)
    grad_cam_dir = plot_grad_cam_samples(
        model=model,
        dataloader=dataloaders['test'],
        target_layer=target_layer,
        class_names=class_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        fold=fold_split['fold_num'],
        device=device
    )
    
    # Print evaluation results
    print(f"\nFold {fold_split['fold_num']} Evaluation:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save metrics
    metrics = {
        'fold': fold_split['fold_num'],
        'best_epoch': best_epoch,
        'train_loss': train_losses[best_epoch],
        'val_loss': val_losses[best_epoch],
        'test_loss': test_loss,
        'train_accuracy': train_accuracies[best_epoch],
        'val_accuracy': best_accuracy,
        'test_accuracy': test_acc,
        'classification_report': report,
        'paths': {
            'best_model': best_model_path,
            'confusion_matrix': cm_path,
            'roc_curves': roc_path,
            'classification_report': report_path,
            'grad_cam_dir': grad_cam_dir
        }
    }
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return best_model_path, metrics


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get fold splits
    fold_splits = get_fold_splits(num_folds=10, fold=args.fold)
    
    # Initialize results
    results = []
    
    # Train and evaluate on each fold
    for fold_split in fold_splits:
        print(f"\n{'='*80}")
        print(f"Training on Fold {fold_split['fold_num']}")
        print(f"Train Folds: {fold_split['train']}")
        print(f"Val Fold: {fold_split['val']}")
        print(f"Test Fold: {fold_split['test']}")
        print(f"{'='*80}\n")
        
        # Train and evaluate model
        best_model_path, metrics = train_model(args, fold_split)
        
        # Collect results
        results.append(metrics)
    
    # Calculate average metrics across folds
    test_accuracies = [r['test_accuracy'] for r in results]
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    
    # Print summary of results
    print(f"\n{'='*80}")
    print(f"Summary of results for {args.model_name}")
    print(f"{'='*80}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.2f}%")
    print(f"Individual Fold Test Accuracies: {test_accuracies}")
    
    # Save summary to file
    summary = {
        'model_name': args.model_name,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'average_test_accuracy': avg_test_accuracy,
        'fold_accuracies': {f"fold_{r['fold']}": r['test_accuracy'] for r in results},
        'args': vars(args),
        'fold_results': results
    }
    
    summary_path = os.path.join(args.output_dir, args.model_name, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")

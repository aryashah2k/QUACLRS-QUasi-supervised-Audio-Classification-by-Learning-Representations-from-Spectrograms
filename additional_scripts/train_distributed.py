"""
Distributed training script for audio spectrogram classification.
This script enables multi-GPU training on the UrbanSound dataset using PyTorch's
distributed data parallel (DDP) functionality.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import json
import time
from datetime import datetime

# Import utilities
from utils.dataset import SpectrogramDataset
from utils.training import train_one_epoch, validate, save_checkpoint
from utils.visualization import (
    plot_learning_curves, 
    plot_confusion_matrix, 
    plot_roc_curves, 
    plot_grad_cam_samples
)
from utils.models import create_model, get_target_layer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Distributed training for audio spectrogram classification')
    
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
    parser.add_argument('--batch_size', type=str, default='32,32',
                        help='Batch size per gpu for training,validation/testing (comma-separated)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading per GPU')
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
    
    # Distributed training parameters
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes for distributed training')
    parser.add_argument('--gpus_per_node', type=int, default=4,
                        help='Number of GPUs per node')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='Ranking of the node for multi-node distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training (set automatically by torch.distributed.launch)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train on (None for all folds)')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Use synchronized batch normalization in distributed training')
    
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


def create_dataloaders(csv_file, base_dir, fold_splits, batch_size, num_workers, local_rank, world_size):
    """
    Create DataLoader objects for distributed training
    
    Args:
        csv_file (str): Path to the CSV file with annotations
        base_dir (str): Directory where spectrogram images are stored
        fold_splits (dict): Dict with keys 'train', 'val', 'test' containing fold numbers
        batch_size (tuple): Tuple of (train_batch_size, eval_batch_size)
        num_workers (int): Number of worker threads for DataLoader per GPU
        local_rank (int): Local rank for distributed training
        world_size (int): Total number of processes in distributed training
        
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoader objects and datasets
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    # Define transformations for training and validation/testing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match common input sizes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    # We don't need distributed sampler for validation and test
    # since we'll gather the results from all processes
    
    # Create dataloaders
    train_batch_size, eval_batch_size = batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # For validation and test, we don't need distributed sampling
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_sampler': train_sampler,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }


def train_model(local_rank, world_size, args, fold_split):
    """
    Train a model on a specific fold split with distributed training
    
    Args:
        local_rank: Local rank for distributed training
        world_size: Total number of processes in distributed training
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
    device = torch.device(f'cuda:{local_rank}')
    
    # Parse batch sizes
    batch_sizes = args.batch_size.split(',')
    train_batch_size = int(batch_sizes[0])
    eval_batch_size = int(batch_sizes[1] if len(batch_sizes) > 1 else batch_sizes[0])
    batch_size = (train_batch_size, eval_batch_size)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        csv_file=args.csv_file,
        base_dir=args.spectrograms_dir,
        fold_splits=fold_split,
        batch_size=batch_size,
        num_workers=args.num_workers,
        local_rank=local_rank,
        world_size=world_size
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
    
    # Convert BatchNorm to SyncBatchNorm for distributed training if requested
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap model with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
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
        # Set epoch for distributed sampler
        dataloaders['train_sampler'].set_epoch(epoch)
        
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
        
        # Gather metrics from all processes
        if world_size > 1:
            # Create tensors for gathering
            train_loss_tensor = torch.tensor([train_loss], device=device)
            val_loss_tensor = torch.tensor([val_loss], device=device)
            train_acc_tensor = torch.tensor([train_acc], device=device)
            val_acc_tensor = torch.tensor([val_acc], device=device)
            
            # Gather all metrics
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
            
            # Average the gathered metrics
            train_loss = train_loss_tensor.item() / world_size
            val_loss = val_loss_tensor.item() / world_size
            train_acc = train_acc_tensor.item() / world_size
            val_acc = val_acc_tensor.item() / world_size
        
        # Save training metrics (only on main process)
        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Save checkpoint and update best model
            current_model_path = save_checkpoint(
                model=model.module,  # Unwrap DDP model for saving
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
    
    # Make sure all processes reach this point
    dist.barrier()
    
    # Generate plots and metrics (only on main process)
    metrics = None
    if local_rank == 0:
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
        model_for_eval = create_model(
            model_name=args.model_name,
            num_classes=num_classes,
            pretrained=False  # We'll load weights from checkpoint
        ).to(device)
        model_for_eval.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        test_loss, test_acc, y_true, y_pred, y_probs = validate(
            model=model_for_eval,
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
        from sklearn.metrics import classification_report
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
        target_layer = get_target_layer(model_for_eval, args.model_name)
        grad_cam_dir = plot_grad_cam_samples(
            model=model_for_eval,
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
    
    # Make sure all processes reach this point
    dist.barrier()
    
    return best_model_path, metrics


def setup(args, rank, world_size):
    """
    Setup the distributed environment
    
    Args:
        args: Command-line arguments
        rank: Global rank of the process
        world_size: Total number of processes
    """
    # Create a temporary directory for file-based initialization
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Initialize the process group with file-based initialization
    dist.init_process_group(
        backend='nccl',
        init_method=f'file://{temp_dir}/shared_file',
        world_size=world_size,
        rank=rank
    )


def cleanup():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()


def run_fold(local_rank, args, fold_split):
    """
    Run training and evaluation for a single fold
    
    Args:
        local_rank: Local rank of the process
        args: Command-line arguments
        fold_split: Dictionary with train, val, and test fold assignments
    """
    # Calculate global rank
    global_rank = args.node_rank * args.gpus_per_node + local_rank
    world_size = args.nodes * args.gpus_per_node
    
    # Setup distributed environment
    setup(args, global_rank, world_size)
    
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Print fold information on main process
    if local_rank == 0:
        print(f"\n{'='*80}")
        print(f"Training on Fold {fold_split['fold_num']} with Distributed Training")
        print(f"Train Folds: {fold_split['train']}")
        print(f"Val Fold: {fold_split['val']}")
        print(f"Test Fold: {fold_split['test']}")
        print(f"Number of processes: {world_size}")
        print(f"{'='*80}\n")
    
    # Train and evaluate model
    best_model_path, metrics = train_model(local_rank, world_size, args, fold_split)
    
    # Clean up
    cleanup()
    
    return metrics


def main_worker():
    """
    Main worker function for distributed training
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set the local rank for this process
    local_rank = args.local_rank
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Create output directory
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get fold splits
    fold_splits = get_fold_splits(num_folds=10, fold=args.fold)
    
    # Initialize results
    results = []
    
    # Train and evaluate on each fold
    for fold_split in fold_splits:
        metrics = run_fold(local_rank, args, fold_split)
        
        # Collect results on main process
        if local_rank == 0 and metrics is not None:
            results.append(metrics)
    
    # Calculate average metrics across folds (on main process only)
    if local_rank == 0 and results:
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
    main_worker()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")

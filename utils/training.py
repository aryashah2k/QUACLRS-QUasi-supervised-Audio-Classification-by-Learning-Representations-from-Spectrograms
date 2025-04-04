import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, scaler=None):
    """
    Train the model for one epoch
    
    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run training on (cuda/cpu)
        epoch: Current epoch number
        scheduler: Learning rate scheduler (optional)
        scaler: GradScaler for mixed precision training (optional)
        
    Returns:
        average_loss: Average loss for the epoch
        accuracy: Training accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for i, (inputs, targets, _) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                # Handle InceptionV3 outputs which return a tuple
                if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                    # If it's InceptionV3 output (has aux_logits)
                    if hasattr(outputs, 'aux_logits'):
                        # Calculate loss as weighted sum of main loss and auxiliary loss
                        main_loss = criterion(outputs.logits, targets)
                        aux_loss = criterion(outputs.aux_logits, targets)
                        loss = main_loss + 0.4 * aux_loss
                    else:
                        loss = criterion(outputs.logits, targets)
                    outputs = outputs.logits
                else:
                    loss = criterion(outputs, targets)
                
            # Scale loss and compute gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs = model(inputs)
            # Handle InceptionV3 outputs which return a tuple
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                # If it's InceptionV3 output (has aux_logits)
                if hasattr(outputs, 'aux_logits'):
                    # Calculate loss as weighted sum of main loss and auxiliary loss
                    main_loss = criterion(outputs.logits, targets)
                    aux_loss = criterion(outputs.aux_logits, targets)
                    loss = main_loss + 0.4 * aux_loss
                else:
                    loss = criterion(outputs.logits, targets)
                outputs = outputs.logits
            else:
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (i + 1),
            'acc': 100. * correct / total
        })
    
    # Apply scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    average_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return average_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Validate the model on validation/test set
    
    Args:
        model: The neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on (cuda/cpu)
        
    Returns:
        average_loss: Average loss for validation
        accuracy: Validation accuracy
        all_targets: All true labels
        all_predictions: All model predictions
        all_probabilities: All prediction probabilities
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, targets, _ in tqdm(dataloader, desc='Validating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            # Handle InceptionV3 outputs which return a tuple
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                if hasattr(outputs, 'aux_logits'):
                    # Use only the main output for validation
                    loss = criterion(outputs.logits, targets)
                    outputs = outputs.logits
                else:
                    loss = criterion(outputs.logits, targets)
                    outputs = outputs.logits
            else:
                loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store targets and predictions for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    average_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return average_loss, accuracy, np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, loss, fold, model_name, output_dir):
    """
    Save model checkpoint
    
    Args:
        model: The neural network model
        optimizer: Optimizer for updating weights
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        accuracy: Current validation accuracy
        loss: Current validation loss
        fold: Current fold number in cross-validation
        model_name: Name of the model architecture
        output_dir: Directory to save checkpoints
    """
    checkpoint_dir = os.path.join(output_dir, model_name, f'fold_{fold}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save regular checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save as best model if it's the best so far
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path)
        if accuracy > best_checkpoint['accuracy']:
            torch.save(checkpoint, best_checkpoint_path)
    else:
        torch.save(checkpoint, best_checkpoint_path)
    
    return checkpoint_path

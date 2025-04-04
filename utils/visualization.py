import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_dir, model_name, fold):
    """
    Plot training and validation learning curves
    
    Args:
        train_losses: List of training losses for each epoch
        val_losses: List of validation losses for each epoch
        train_accuracies: List of training accuracies for each epoch
        val_accuracies: List of validation accuracies for each epoch
        output_dir: Directory to save plots
        model_name: Name of the model architecture
        fold: Current fold number in cross-validation
    """
    save_dir = os.path.join(output_dir, model_name, f'fold_{fold}', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Fold {fold} - Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Fold {fold} - Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
    plt.close()
    
    return save_dir


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, model_name, fold, normalize=True):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save plots
        model_name: Name of the model architecture
        fold: Current fold number in cross-validation
        normalize: Whether to normalize the confusion matrix
    """
    save_dir = os.path.join(output_dir, model_name, f'fold_{fold}', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Fold {fold} - Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    return os.path.join(save_dir, 'confusion_matrix.png')


def plot_roc_curves(y_true, y_scores, class_names, output_dir, model_name, fold):
    """
    Plot ROC curves for each class
    
    Args:
        y_true: True labels (one-hot encoded)
        y_scores: Predicted probabilities
        class_names: List of class names
        output_dir: Directory to save plots
        model_name: Name of the model architecture
        fold: Current fold number in cross-validation
    """
    save_dir = os.path.join(output_dir, model_name, f'fold_{fold}', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(class_names)
    
    # One-hot encode the labels for ROC curve calculation
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, val in enumerate(y_true):
        y_true_onehot[i, val] = 1
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Fold {fold} - ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()
    
    return os.path.join(save_dir, 'roc_curves.png')


def generate_grad_cam(model, input_tensor, target_layer, class_idx, original_image):
    """
    Generate Grad-CAM visualization for a single image
    
    Args:
        model: The neural network model
        input_tensor: Preprocessed input tensor
        target_layer: Target layer for Grad-CAM
        class_idx: Target class index
        original_image: Original image for visualization
        
    Returns:
        visualization: Grad-CAM visualization
    """
    # Create GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Specify the target class
    targets = [ClassifierOutputTarget(class_idx)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Convert original image to numpy if it's a tensor
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy().transpose(1, 2, 0)
    
    # Normalize the image for visualization
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    
    # Create visualization
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    
    return visualization


def plot_grad_cam_samples(model, dataloader, target_layer, class_names, output_dir, model_name, fold, device, num_samples=5):
    """
    Plot Grad-CAM visualizations for sample images
    
    Args:
        model: The neural network model
        dataloader: DataLoader containing samples
        target_layer: Target layer for Grad-CAM
        class_names: List of class names
        output_dir: Directory to save plots
        model_name: Name of the model architecture
        fold: Current fold number in cross-validation
        device: Device to run model on (cuda/cpu)
        num_samples: Number of samples to visualize per class
    """
    save_dir = os.path.join(output_dir, model_name, f'fold_{fold}', 'grad_cam')
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples_per_class = {i: 0 for i in range(len(class_names))}
    processed_samples = 0
    
    for inputs, targets, filenames in dataloader:
        # Break when we have enough samples for each class
        if min(samples_per_class.values()) >= num_samples:
            break
            
        for i in range(len(targets)):
            target = targets[i].item()
            
            # Skip if we already have enough samples for this class
            if samples_per_class[target] >= num_samples:
                continue
                
            # Prepare input for GradCAM
            input_tensor = inputs[i:i+1].to(device)
            
            # Get model prediction
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = output.max(1)
                
            predicted_class = predicted.item()
            
            # Generate GradCAM for the predicted class
            original_image = inputs[i].cpu().numpy().transpose(1, 2, 0)
            
            # Normalize the original image for visualization
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
            
            # Generate Grad-CAM
            cam = GradCAM(model=model, target_layers=[target_layer])
            # Specify the target class
            cam_targets = [ClassifierOutputTarget(predicted_class)]
            # Generate CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=cam_targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Create visualization
            visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
            
            # Plot and save
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title(f'Original: {class_names[target]}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(visualization)
            plt.title(f'Grad-CAM: {class_names[predicted_class]}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'gradcam_{class_names[target]}_{samples_per_class[target]}.png'))
            plt.close()
            
            samples_per_class[target] += 1
            processed_samples += 1
            
    return save_dir

import torch
import torch.nn as nn
import torchvision.models as models

def create_alexnet(num_classes, pretrained=True, freeze_features=False):
    """
    Create an AlexNet model for spectrogram classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: Modified AlexNet model
    """
    if pretrained:
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    else:
        model = models.alexnet(weights=None)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Modify the classifier for our task
    model.classifier[6] = nn.Linear(4096, num_classes)
    
    return model


def create_vgg16(num_classes, pretrained=True, freeze_features=False):
    """
    Create a VGG16 model for spectrogram classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: Modified VGG16 model
    """
    if pretrained:
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    else:
        model = models.vgg16(weights=None)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Modify the classifier for our task
    model.classifier[6] = nn.Linear(4096, num_classes)
    
    return model


def create_resnet18(num_classes, pretrained=True, freeze_features=False):
    """
    Create a ResNet18 model for spectrogram classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: Modified ResNet18 model
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    # Modify the final fully connected layer for our task
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def create_mobilenet(num_classes, pretrained=True, freeze_features=False):
    """
    Create a MobileNetV2 model for spectrogram classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: Modified MobileNetV2 model
    """
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        model = models.mobilenet_v2(weights=None)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    # Modify the classifier for our task
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model


def create_inception(num_classes, pretrained=True, freeze_features=False):
    """
    Create an InceptionV3 model for spectrogram classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: Modified InceptionV3 model
    """
    if pretrained:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    else:
        model = models.inception_v3(weights=None, aux_logits=True)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for name, param in model.named_parameters():
            if "fc" not in name and "AuxLogits" not in name:
                param.requires_grad = False
    
    # Modify the fully connected layers for our task
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    
    return model


def create_efficientnet(num_classes, pretrained=True, freeze_features=False):
    """
    Create an EfficientNet-B0 model for spectrogram classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: Modified EfficientNet-B0 model
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    # Modify the classifier for our task
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model


def create_convnext(num_classes, pretrained=True, freeze_features=False):
    """
    Create a ConvNeXt-Tiny model for spectrogram classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: Modified ConvNeXt-Tiny model
    """
    if pretrained:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    else:
        model = models.convnext_tiny(weights=None)
    
    # Freeze feature extraction layers if specified
    if freeze_features:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    # Modify the classifier for our task
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    return model


def get_target_layer(model, model_name):
    """
    Get the target layer for Grad-CAM visualization based on model architecture
    
    Args:
        model: The neural network model
        model_name: Name of the model architecture
        
    Returns:
        target_layer: The target layer for Grad-CAM
    """
    # For each architecture, return the appropriate layer for GradCAM
    if model_name == 'alexnet':
        # Check if features are frozen
        for param in model.features[-1].parameters():
            if param.requires_grad:
                return model.features[-1]
        # If frozen, use classifier
        return model.classifier[0]
    elif model_name == 'vgg16':
        # Check if features are frozen
        for param in model.features[-1].parameters():
            if param.requires_grad:
                return model.features[-1]
        # If frozen, use classifier
        return model.classifier[0]
    elif model_name == 'resnet18':
        # Check if layer4 is frozen
        for param in model.layer4[-1].parameters():
            if param.requires_grad:
                return model.layer4[-1]
        # If frozen, use earlier layer with gradients or avgpool
        return model.avgpool
    elif model_name == 'mobilenet':
        # For MobileNet, check if the features are frozen
        # If frozen, we need to use an earlier layer that has gradients
        for param in model.features[-1].parameters():
            if param.requires_grad:
                return model.features[-1]
        # If all params in the last layer are frozen, use one of the classifier layers
        return model.classifier[0]
    elif model_name == 'inception':
        # For InceptionV3, use the last Mixed layer before AuxLogits
        # Check if it's frozen
        for param in model.Mixed_7c.parameters():
            if param.requires_grad:
                return model.Mixed_7c
        # If frozen, use AuxLogits (which should have gradients)
        return model.AuxLogits.conv0
    elif model_name == 'efficientnet':
        # Check if features are frozen
        for param in model.features[-1].parameters():
            if param.requires_grad:
                return model.features[-1]
        # If frozen, use classifier
        return model.classifier[0]
    elif model_name == 'convnext':
        # Check if features are frozen
        for param in model.features[-1].parameters():
            if param.requires_grad:
                return model.features[-1]
        # If frozen, use classifier
        return model.classifier[0]
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_model(model_name, num_classes, pretrained=True, freeze_features=False):
    """
    Create a model based on the specified architecture
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extraction layers
        
    Returns:
        model: The neural network model
    """
    if model_name == 'alexnet':
        return create_alexnet(num_classes, pretrained, freeze_features)
    elif model_name == 'vgg16':
        return create_vgg16(num_classes, pretrained, freeze_features)
    elif model_name == 'resnet18':
        return create_resnet18(num_classes, pretrained, freeze_features)
    elif model_name == 'mobilenet':
        return create_mobilenet(num_classes, pretrained, freeze_features)
    elif model_name == 'inception':
        return create_inception(num_classes, pretrained, freeze_features)
    elif model_name == 'efficientnet':
        return create_efficientnet(num_classes, pretrained, freeze_features)
    elif model_name == 'convnext':
        return create_convnext(num_classes, pretrained, freeze_features)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

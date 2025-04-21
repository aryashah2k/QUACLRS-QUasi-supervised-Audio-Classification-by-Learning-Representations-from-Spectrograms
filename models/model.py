import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import io
import base64

# Model Components (matching your original implementation)
class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
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
    def __init__(self, input_dim=512, num_classes=13):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

def process_audio(audio_data, sr=22050, duration=None, n_mels=224):
    """Process audio to spectrogram (matches training config)"""
    # Handle both file paths and audio data
    if isinstance(audio_data, str):
        # It's a file path
        y, sr = librosa.load(audio_data, sr=sr, duration=duration)
    else:
        # It's already loaded audio data
        y = audio_data
        
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, 
                                     hop_length=512, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Create a figure for the spectrogram
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to a temporary buffer instead of a file
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()
    buf.seek(0)
    
    img = Image.open(buf).convert('RGB')
    return img, y, sr, S_dB  # Return S_dB for additional visualizations

class AudioClassifier:
    def __init__(self, checkpoint_path, class_names, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # MobileNetV3-Small backbone matching training
        from torchvision.models import mobilenet_v3_small
        backbone = mobilenet_v3_small(weights=None)
        
        # Disable inplace operations for XAI compatibility
        for module in backbone.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
                
        backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(576, 512),
            torch.nn.Hardswish(inplace=False)  # Explicitly disable inplace
        )

        self.backbone = backbone
        self.simclr = SimCLR(backbone).to(self.device)
        self.classifier = Classifier().to(self.device)
        self.class_names = class_names

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.simclr.load_state_dict(checkpoint['simclr'], strict=False)
        self.classifier.load_state_dict(checkpoint['classifier'], strict=False)

        self.simclr.eval()
        self.classifier.eval()

        # Input transforms matching training
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        # Set up hooks for GradCAM
        self.activations = None
        self.gradients = None
        self.setup_gradcam_hooks()
        
    def setup_gradcam_hooks(self):
        """Set up hooks for GradCAM visualization"""
        # Target the last convolutional layer in MobileNetV3
        target_layer = self.backbone.features[-1][-1]
        
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate_gradcam(self, img_tensor, target_class=None):
        """Generate GradCAM visualization for the input image"""
        # Forward pass
        self.simclr.zero_grad()
        features = self.simclr.backbone(img_tensor)
        logits = self.classifier(features)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass to get gradients
        logits[0, target_class].backward(retain_graph=True)
        
        # Compute weights using gradients
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weight the activations by the gradients
        heatmap = torch.sum(pooled_gradients * self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)  # ReLU to only show positive contributions
        
        # Normalize heatmap
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy(), target_class
    
    def compute_integrated_gradients(self, img_tensor, target_class=None, steps=50):
        """Compute Integrated Gradients for the input image"""
        # Create baseline (black image)
        baseline = torch.zeros_like(img_tensor).to(self.device)
        
        # Forward pass to get prediction
        self.simclr.zero_grad()
        features = self.simclr.backbone(img_tensor)
        logits = self.classifier(features)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Compute integrated gradients
        integrated_grads = torch.zeros_like(img_tensor).to(self.device)
        
        for step in range(1, steps + 1):
            # Interpolate between baseline and input
            interpolated = baseline + (img_tensor - baseline) * (step / steps)
            interpolated.requires_grad = True
            
            # Forward pass
            features = self.simclr.backbone(interpolated)
            logits = self.classifier(features)
            
            # Backward pass
            self.simclr.zero_grad()
            logits[0, target_class].backward(retain_graph=True)
            
            # Get gradients
            gradients = interpolated.grad.detach()
            
            # Add to integrated gradients
            integrated_grads += gradients / steps
        
        # Multiply by input - baseline
        integrated_grads *= (img_tensor - baseline)
        
        # Sum across color channels for visualization
        attribution_map = torch.sum(torch.abs(integrated_grads), dim=1).squeeze()
        
        # Normalize for visualization
        if torch.max(attribution_map) > 0:
            attribution_map /= torch.max(attribution_map)
            
        return attribution_map.cpu().numpy(), target_class
    
    def predict(self, audio_input, verbose=True, visualize_xai=False):
        """Predict class for audio file with XAI visualizations"""
        if verbose and isinstance(audio_input, str):
            print(f"Processing audio: {audio_input}")
            
        # Process audio to spectrogram
        img, audio, sr, S_dB = process_audio(audio_input)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            features = self.simclr.backbone(img_tensor)
            logits = self.classifier(features)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Print top predictions if verbose
        if verbose:
            top_indices = np.argsort(probs)[::-1][:3]
            print("\nTop 3 predictions:")
            for i, idx in enumerate(top_indices):
                print(f"{i+1}. {self.class_names[idx]}: {probs[idx]*100:.2f}%")
        
        # Return visualization data for the Streamlit app
        if visualize_xai:
            viz_data = self.generate_visualization_data(img, img_tensor, probs, S_dB)
            return probs, audio, sr, viz_data
            
        return probs, audio, sr, None
    
    def generate_visualization_data(self, original_img, img_tensor, probs, spectrogram_db):
        """Generate visualization data for Streamlit display"""
        # Get top predicted class
        top_class = np.argmax(probs)
        
        # 1. GradCAM visualization
        gradcam_heatmap, _ = self.generate_gradcam(img_tensor, top_class)
        
        # 2. Integrated Gradients visualization
        ig_heatmap, _ = self.compute_integrated_gradients(img_tensor, top_class)
        
        # Resize heatmaps to match original image
        original_size = (original_img.width, original_img.height)
        gradcam_heatmap_resized = cv2.resize(gradcam_heatmap, original_size)
        ig_heatmap_resized = cv2.resize(ig_heatmap, original_size)
        
        # Convert heatmaps to RGB for visualization
        gradcam_heatmap_rgb = cv2.applyColorMap(np.uint8(255 * gradcam_heatmap_resized), cv2.COLORMAP_JET)
        gradcam_heatmap_rgb = cv2.cvtColor(gradcam_heatmap_rgb, cv2.COLOR_BGR2RGB)
        
        # Create overlay images
        original_img_np = np.array(original_img)
        gradcam_overlay = cv2.addWeighted(original_img_np, 0.6, gradcam_heatmap_rgb, 0.4, 0)
        
        # Return all visualization data
        return {
            'original_img': original_img,
            'gradcam_heatmap': gradcam_heatmap_resized,
            'gradcam_overlay': gradcam_overlay,
            'ig_heatmap': ig_heatmap_resized,
            'spectrogram_db': spectrogram_db,
            'top_class': top_class,
            'probs': probs
        }
    
    def get_figure_as_image(self):
        """Helper function to convert a matplotlib figure to an image"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

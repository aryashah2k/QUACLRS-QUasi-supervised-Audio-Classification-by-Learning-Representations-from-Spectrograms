import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import IPython.display as ipd
import cv2

# Model Components (updated for MobileNetV3)
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

def process_audio(audio_path, sr=22050, duration=None, n_mels=224):
    """Process audio to spectrogram (matches training config)"""
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, 
                                      hop_length=512, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    temp_path = 'temp_spec.png'
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    img = Image.open(temp_path).convert('RGB')
    if os.path.exists(temp_path):
        os.remove(temp_path)
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
    
    def predict(self, audio_path, verbose=True, visualize_xai=True):
        """Predict class for audio file with XAI visualizations"""
        if verbose:
            print(f"Processing audio: {audio_path}")
            
        # Process audio to spectrogram
        img, audio, sr, S_dB = process_audio(audio_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            features = self.simclr.backbone(img_tensor)
            logits = self.classifier(features)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Print top predictions
        if verbose:
            top_indices = np.argsort(probs)[::-1][:3]
            print("\nTop 3 predictions:")
            for i, idx in enumerate(top_indices):
                print(f"{i+1}. {self.class_names[idx]}: {probs[idx]*100:.2f}%")
        
        # Generate XAI visualizations
        if visualize_xai:
            self.visualize_xai(img, img_tensor, probs, S_dB)
            
        return probs, audio, sr

    def visualize_xai(self, original_img, img_tensor, probs, spectrogram_db):
        """Generate and display multiple XAI visualizations"""
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
        
        # Create figure for visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Original spectrogram
        plt.subplot(2, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Spectrogram")
        plt.axis('off')
        
        # 2. GradCAM heatmap
        plt.subplot(2, 3, 2)
        plt.imshow(gradcam_heatmap_resized, cmap='jet')
        plt.title(f"GradCAM: {self.class_names[top_class]}")
        plt.axis('off')
        
        # 3. GradCAM overlay
        plt.subplot(2, 3, 3)
        plt.imshow(gradcam_overlay)
        plt.title("GradCAM Overlay")
        plt.axis('off')
        
        # 4. Integrated Gradients heatmap
        plt.subplot(2, 3, 4)
        plt.imshow(ig_heatmap_resized, cmap='viridis')
        plt.title("Integrated Gradients")
        plt.axis('off')
        
        # 5. Frequency-Time Analysis
        plt.subplot(2, 3, 5)
        plt.imshow(spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram (dB)")
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Frequency Bands')
        
        # 6. Class Probability Distribution
        plt.subplot(2, 3, 6)
        top_indices = np.argsort(probs)[::-1][:5]  # Show top 5 classes
        plt.barh([self.class_names[i] for i in top_indices], 
                 [probs[i] for i in top_indices])
        plt.title("Top 5 Class Probabilities")
        plt.xlabel("Probability")
        
        plt.tight_layout()
        plt.show()

# Visualization function remains unchanged
def visualize_prediction(probs, audio, sr, class_names):
    print("Audio sample:")
    display(ipd.Audio(audio, rate=sr))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.subplot(1, 2, 2)
    indices = np.argsort(probs)[::-1]
    plt.barh(range(len(class_names)), [probs[i] for i in indices])
    plt.yticks(range(len(class_names)), [class_names[i] for i in indices])
    plt.title("Class Probabilities")
    plt.xlabel("Probability")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 
                  'dog_bark', 'drilling', 'engine_idling', 
                  'gun_shot', 'jackhammer', 'ambulance', 'firetruck', 'police', 'traffic', 'street_music']
    classifier = AudioClassifier(
        checkpoint_path='fold_10_checkpoint.pth',
        class_names=class_names
    )
    audio_path = "./TestAudioFiles/starwars.wav"
    probs, audio, sr = classifier.predict(audio_path)
    visualize_prediction(probs, audio, sr, class_names)
    top_idx = np.argmax(probs)
    print(f"\nFinal prediction: {class_names[top_idx]} with {probs[top_idx]*100:.2f}% confidence")

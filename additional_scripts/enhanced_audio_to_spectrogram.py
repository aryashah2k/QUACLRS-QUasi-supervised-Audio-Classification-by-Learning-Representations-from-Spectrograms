#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Audio-to-Spectrogram Converter with Augmentation Techniques

This script enhances the original Audio-to-Spectrogram.py script with multiple
augmentation techniques specifically for spectrograms, including:
- SpecAugment (time/frequency masking)
- PatchAugment (randomly replace patches)
- Time stretching (0.8x-1.2x)
- Pitch shifting (±2 semitones)
- Gaussian noise addition (5-15dB SNR)

The script can be used standalone or as part of the complete dataset extension pipeline.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import random
import argparse
from tqdm import tqdm
import warnings
import soundfile as sf
import uuid
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Parameters for mel spectrogram generation
SR = 22050  # Sample rate
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
N_MELS = 128  # Number of mel bins
FMIN = 20  # Min frequency
FMAX = 8000  # Max frequency
DURATION = 4.0  # Target duration in seconds

class AudioAugmenter:
    """Class for audio augmentation and spectrogram generation."""
    
    def __init__(self, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX, duration=DURATION):
        """Initialize the augmenter with spectrogram parameters."""
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
    
    def load_audio(self, file_path):
        """Load audio file and standardize to target duration."""
        try:
            # Load audio file
            y, orig_sr = librosa.load(file_path, sr=None)
            
            # Resample if needed
            if orig_sr != self.sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
            
            # Adjust duration
            target_length = int(self.duration * self.sr)
            if len(y) > target_length:
                # Trim to target length
                y = y[:target_length]
            elif len(y) < target_length:
                # Pad with zeros
                y = np.pad(y, (0, target_length - len(y)), 'constant')
            
            return y
        except Exception as e:
            print(f"Error loading audio {file_path}: {str(e)}")
            return None
    
    def create_mel_spectrogram(self, y):
        """Convert audio to mel spectrogram."""
        if y is None:
            return None
        
        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax
        )
        
        # Convert to dB scale
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        return S_dB
    
    def time_stretch(self, y, rate):
        """Apply time stretching augmentation."""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(self, y, n_steps):
        """Apply pitch shifting augmentation."""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
    
    def add_noise(self, y, snr_db):
        """Add Gaussian noise with specified SNR."""
        # Calculate signal power
        signal_power = np.mean(y ** 2)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        
        # Add noise to signal
        return y + noise
    
    def spec_augment(self, spec, num_time_masks=2, num_freq_masks=2, 
                    max_time_mask=20, max_freq_mask=20):
        """Apply SpecAugment (time/frequency masking)."""
        spec_aug = spec.copy()
        
        # Apply time masking
        for _ in range(num_time_masks):
            t_start = np.random.randint(0, spec.shape[1] - max_time_mask)
            t_width = np.random.randint(1, max_time_mask)
            spec_aug[:, t_start:t_start+t_width] = np.min(spec)
        
        # Apply frequency masking
        for _ in range(num_freq_masks):
            f_start = np.random.randint(0, spec.shape[0] - max_freq_mask)
            f_width = np.random.randint(1, max_freq_mask)
            spec_aug[f_start:f_start+f_width, :] = np.min(spec)
        
        return spec_aug
    
    def patch_augment(self, spec, num_patches=5, patch_size=(10, 10)):
        """Apply PatchAugment (randomly replace patches)."""
        spec_aug = spec.copy()
        h, w = spec.shape
        patch_h, patch_w = patch_size
        
        # Choose random patch replacement areas
        for _ in range(num_patches):
            # Ensure we don't exceed image boundaries
            if h <= patch_h or w <= patch_w:
                continue
                
            # Source patch position
            src_h = np.random.randint(0, h - patch_h)
            src_w = np.random.randint(0, w - patch_w)
            
            # Target patch position
            tgt_h = np.random.randint(0, h - patch_h)
            tgt_w = np.random.randint(0, w - patch_w)
            
            # Copy patch from source to target
            spec_aug[tgt_h:tgt_h+patch_h, tgt_w:tgt_w+patch_w] = \
                spec[src_h:src_h+patch_h, src_w:src_w+patch_w].copy()
        
        return spec_aug
    
    def save_spectrogram(self, spec, output_path, figsize=(3, 3), dpi=100):
        """Save spectrogram as an image file."""
        plt.figure(figsize=figsize, dpi=dpi)
        plt.axis('off')
        librosa.display.specshow(
            spec, sr=self.sr, hop_length=self.hop_length,
            x_axis=None, y_axis=None
        )
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def apply_augmentations(self, audio_path, output_dir, augment_config=None):
        """
        Apply all augmentation techniques to an audio file and save spectrograms.
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        output_dir : str
            Directory to save spectrograms
        augment_config : dict
            Configuration for augmentations with keys:
            - time_stretch: list of rates
            - pitch_shift: list of semitones
            - noise: list of SNR values in dB
            - spec_augment: number of variants
            - patch_augment: number of variants
            
        Returns:
        --------
        list of dict
            Metadata for all generated spectrograms
        """
        # Default augmentation config
        if augment_config is None:
            augment_config = {
                'time_stretch': [0.8, 1.2],
                'pitch_shift': [-2, 2],
                'noise': [5, 15],
                'spec_augment': 3,
                'patch_augment': 2
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        y = self.load_audio(audio_path)
        if y is None:
            return []
        
        # Extract file info for naming
        file_name = os.path.basename(audio_path)
        file_base = os.path.splitext(file_name)[0]
        
        # Create base mel spectrogram
        base_spec = self.create_mel_spectrogram(y)
        if base_spec is None:
            return []
        
        # Initialize metadata list
        metadata = []
        
        # Save original spectrogram
        orig_spec_name = f"{file_base}_orig.png"
        orig_spec_path = os.path.join(output_dir, orig_spec_name)
        self.save_spectrogram(base_spec, orig_spec_path)
        
        # Add to metadata
        metadata.append({
            'spec_file_name': orig_spec_name,
            'orig_file_name': file_name,
            'augmentation': 'original'
        })
        
        # Apply time stretching
        for i, rate in enumerate(augment_config['time_stretch']):
            y_stretched = self.time_stretch(y, rate)
            spec = self.create_mel_spectrogram(y_stretched)
            if spec is None:
                continue
                
            spec_name = f"{file_base}_ts{i}.png"
            spec_path = os.path.join(output_dir, spec_name)
            self.save_spectrogram(spec, spec_path)
            
            metadata.append({
                'spec_file_name': spec_name,
                'orig_file_name': file_name,
                'augmentation': f'timestretch_{rate}'
            })
        
        # Apply pitch shifting
        for i, n_steps in enumerate(augment_config['pitch_shift']):
            y_shifted = self.pitch_shift(y, n_steps)
            spec = self.create_mel_spectrogram(y_shifted)
            if spec is None:
                continue
                
            spec_name = f"{file_base}_ps{i}.png"
            spec_path = os.path.join(output_dir, spec_name)
            self.save_spectrogram(spec, spec_path)
            
            metadata.append({
                'spec_file_name': spec_name,
                'orig_file_name': file_name,
                'augmentation': f'pitchshift_{n_steps}'
            })
        
        # Apply noise addition
        for i, snr in enumerate(augment_config['noise']):
            y_noisy = self.add_noise(y, snr)
            spec = self.create_mel_spectrogram(y_noisy)
            if spec is None:
                continue
                
            spec_name = f"{file_base}_n{i}.png"
            spec_path = os.path.join(output_dir, spec_name)
            self.save_spectrogram(spec, spec_path)
            
            metadata.append({
                'spec_file_name': spec_name,
                'orig_file_name': file_name,
                'augmentation': f'noise_{snr}dB'
            })
        
        # Apply SpecAugment
        for i in range(augment_config['spec_augment']):
            spec_augmented = self.spec_augment(base_spec)
            spec_name = f"{file_base}_sa{i}.png"
            spec_path = os.path.join(output_dir, spec_name)
            self.save_spectrogram(spec_augmented, spec_path)
            
            metadata.append({
                'spec_file_name': spec_name,
                'orig_file_name': file_name,
                'augmentation': f'specaugment_{i}'
            })
        
        # Apply PatchAugment
        for i in range(augment_config['patch_augment']):
            patch_augmented = self.patch_augment(base_spec)
            spec_name = f"{file_base}_pa{i}.png"
            spec_path = os.path.join(output_dir, spec_name)
            self.save_spectrogram(patch_augmented, spec_path)
            
            metadata.append({
                'spec_file_name': spec_name,
                'orig_file_name': file_name,
                'augmentation': f'patchaugment_{i}'
            })
        
        return metadata

def process_dataset(audio_dir, output_dir, metadata_path=None, recursive=True):
    """
    Process an entire audio dataset, generating spectrograms with augmentations.
    
    Parameters:
    -----------
    audio_dir : str
        Directory containing audio files
    output_dir : str
        Directory to save spectrograms
    metadata_path : str, optional
        Path to save metadata CSV
    recursive : bool
        Whether to search for audio files recursively
    """
    # Create augmenter
    augmenter = AudioAugmenter()
    
    # Find all audio files
    audio_files = []
    if recursive:
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(root, file))
    else:
        audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
                      if os.path.isfile(os.path.join(audio_dir, f)) and
                      f.lower().endswith(('.wav', '.mp3'))]
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    all_metadata = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Create relative subdirectory structure if needed
        if recursive:
            rel_path = os.path.relpath(os.path.dirname(audio_file), audio_dir)
            file_output_dir = os.path.join(output_dir, rel_path)
        else:
            file_output_dir = output_dir
        
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Apply augmentations
        try:
            metadata = augmenter.apply_augmentations(audio_file, file_output_dir)
            all_metadata.extend(metadata)
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Save metadata
    if metadata_path and all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
    
    print("Processing complete!")

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Generate mel spectrograms with augmentations from audio files')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input directory containing audio files')
    parser.add_argument('--output', '-o', type=str, required=True,
                      help='Output directory for spectrograms')
    parser.add_argument('--metadata', '-m', type=str,
                      help='Path to save metadata CSV')
    parser.add_argument('--recursive', '-r', action='store_true',
                      help='Search for audio files recursively')
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.metadata, args.recursive)

if __name__ == "__main__":
    main()

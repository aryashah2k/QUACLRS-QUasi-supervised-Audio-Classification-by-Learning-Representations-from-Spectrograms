#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UrbanSound8K Dataset Extension Script

This script extends the UrbanSound8K dataset by:
1. Adding new audio classes from the sireNNet directory
2. Updating the metadata CSV file
3. Converting all audio files to mel spectrograms with augmentations
4. Implementing class balancing
5. Saving the extended dataset
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import shutil
import warnings
import cv2
from sklearn.model_selection import train_test_split
from scipy.signal import resample
import uuid
import time
import math
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants and configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
US8K_DIR = os.path.join(BASE_DIR, 'urbansound8k')
SIRENET_DIR = os.path.join(BASE_DIR, 'sireNNet')
US8K_CSV = os.path.join(US8K_DIR, 'UrbanSound8K.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'urbansound_extended')
SPEC_DIR = os.path.join(OUTPUT_DIR, 'spectrograms')
SPEC_CSV = os.path.join(OUTPUT_DIR, 'spectrograms.csv')

# Mel spectrogram parameters
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000
DURATION = 4  # seconds
SR = 48000  # Sample rate

# Augmentation parameters
AUGMENTATION_PER_CLASS = {
    'time_stretch': 2,  # Number of time stretching variants (0.8x, 1.2x)
    'pitch_shift': 2,   # Number of pitch shifting variants (-2, +2 semitones)
    'spec_augment': 3,  # Number of SpecAugment variants
    'patch_augment': 2, # Number of PatchAugment variants
    'noise': 2,         # Number of noise variants (5dB, 15dB SNR)
}

# Class balancing parameters
TARGET_RATIO = 0.5  # 50/50 class distribution (majority/minority)
TARGET_SAMPLES_PER_FOLD = 150  # Target number of samples per class per fold

class UrbanSoundExtender:
    def __init__(self):
        """Initialize the dataset extender."""
        self.original_metadata = None
        self.new_metadata = pd.DataFrame()
        self.class_mapping = {}
        self.class_counts = {}
        self.next_class_id = 0
        self.label_counts = None
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for fold in range(1, 11):
            os.makedirs(os.path.join(SPEC_DIR, f'fold{fold}'), exist_ok=True)
        
    def load_metadata(self):
        """Load and analyze the original UrbanSound8K metadata."""
        print("Loading UrbanSound8K metadata...")
        self.original_metadata = pd.read_csv(US8K_CSV)
        
        # Analyze class distribution
        self.label_counts = self.original_metadata['classID'].value_counts()
        self.next_class_id = self.original_metadata['classID'].max() + 1
        
        # Create class mapping (original classes)
        for idx, class_name in zip(self.original_metadata['classID'].unique(), 
                                   self.original_metadata['class'].unique()):
            self.class_mapping[class_name] = int(idx)
            self.class_counts[class_name] = self.original_metadata[
                self.original_metadata['class'] == class_name].shape[0]
        
        print(f"Original dataset has {len(self.original_metadata)} samples across "
              f"{len(self.class_mapping)} classes")
        print(f"Class distribution: {self.class_counts}")
    
    def analyze_sirenet_data(self):
        """Analyze the new audio files from the sireNNet directory."""
        print("Analyzing sireNNet data...")
        new_classes = os.listdir(SIRENET_DIR)
        
        for class_name in new_classes:
            class_dir = os.path.join(SIRENET_DIR, class_name)
            if os.path.isdir(class_dir):
                # Assign new class ID
                self.class_mapping[class_name] = self.next_class_id
                self.next_class_id += 1
                
                # Count files
                files = [f for f in os.listdir(class_dir) 
                         if os.path.isfile(os.path.join(class_dir, f)) and 
                         f.lower().endswith(('.wav', '.mp3'))]
                self.class_counts[class_name] = len(files)
        
        print(f"New classes: {[c for c in self.class_mapping if c not in self.original_metadata['class'].unique()]}")
        print(f"Updated class distribution: {self.class_counts}")
    
    def integrate_new_files(self):
        """Integrate new audio files into the existing fold structure."""
        print("Integrating new audio files into fold structure...")
        new_files_info = []
        
        # Process new classes
        new_classes = [c for c in self.class_mapping 
                      if c not in self.original_metadata['class'].unique()]
        
        for class_name in new_classes:
            class_dir = os.path.join(SIRENET_DIR, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_id = self.class_mapping[class_name]
            audio_files = [f for f in os.listdir(class_dir) 
                          if os.path.isfile(os.path.join(class_dir, f)) and 
                          f.lower().endswith(('.wav', '.mp3'))]
            
            # Stratified distribution of files across 10 folds
            files_per_fold = {}
            for fold in range(1, 11):
                files_per_fold[fold] = []
            
            # Distribute files evenly across folds
            for i, file in enumerate(audio_files):
                fold = (i % 10) + 1
                files_per_fold[fold].append(file)
            
            # Create metadata entries and copy files
            for fold, files in files_per_fold.items():
                for idx, file in enumerate(files):
                    # Generate a unique fsID (using timestamp + random)
                    fs_id = int(str(int(time.time()*1000))[-7:] + str(random.randint(100, 999)))
                    
                    # Create new file name following the UrbanSound8K pattern
                    new_file_name = f"{fs_id}-{class_id}-0-{idx}.wav"
                    source_path = os.path.join(class_dir, file)
                    dest_path = os.path.join(US8K_DIR, f"fold{fold}", new_file_name)
                    
                    # Copy file (for real implementation)
                    try:
                        shutil.copy2(source_path, dest_path)
                        
                        # Add to new metadata
                        new_files_info.append({
                            "slice_file_name": new_file_name,
                            "fsID": fs_id,
                            "start": 0.0,
                            "end": 4.0,  # Assuming 4s duration
                            "salience": 1,
                            "fold": fold,
                            "classID": class_id,
                            "class": class_name
                        })
                    except Exception as e:
                        print(f"Error copying {file}: {str(e)}")
        
        # Create DataFrame for new files
        self.new_metadata = pd.DataFrame(new_files_info)
        
        # Merge with original metadata
        self.updated_metadata = pd.concat([self.original_metadata, self.new_metadata], 
                                         ignore_index=True)
        
        # Save updated metadata
        updated_csv_path = os.path.join(OUTPUT_DIR, 'UrbanSound8K_extended.csv')
        self.updated_metadata.to_csv(updated_csv_path, index=False)
        print(f"Updated metadata saved to {updated_csv_path}")
        print(f"Added {len(self.new_metadata)} new files from {len(new_classes)} new classes")
    
    def load_and_preprocess_audio(self, audio_path, duration=DURATION, sr=SR):
        """Load and preprocess audio file."""
        try:
            # Load audio with resampling
            y, orig_sr = librosa.load(audio_path, sr=None)
            
            # Resample if needed
            if orig_sr != sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
            
            # Adjust length to target duration
            target_length = int(duration * sr)
            if len(y) > target_length:
                # Trim to target length
                y = y[:target_length]
            elif len(y) < target_length:
                # Pad with zeros to target length
                y = np.pad(y, (0, target_length - len(y)), 'constant')
            
            return y, sr
        except Exception as e:
            print(f"Error loading audio {audio_path}: {str(e)}")
            return None, None
    
    def create_mel_spectrogram(self, y, sr=SR):
        """Convert audio to mel spectrogram."""
        if y is None:
            return None
            
        try:
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX
            )
            
            # Convert to dB scale
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            return S_dB
        except Exception as e:
            print(f"Error creating mel spectrogram: {str(e)}")
            return None
    
    def time_stretch(self, y, rate):
        """Time stretch augmentation."""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(self, y, sr, n_steps):
        """Pitch shift augmentation."""
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
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
    
    def save_spectrogram_image(self, spec, save_path):
        """Save spectrogram as an image."""
        plt.figure(figsize=(10, 4), dpi=100)
        plt.axis('off')
        librosa.display.specshow(spec, sr=SR, hop_length=HOP_LENGTH, 
                                x_axis=None, y_axis=None)
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def generate_spectrograms(self):
        """Generate mel spectrograms with augmentations for all audio files."""
        print("Generating mel spectrograms with augmentations...")
        
        # Create a DataFrame to store spectrogram metadata
        spec_metadata = []
        
        # Process all files from the updated metadata
        for _, row in tqdm(self.updated_metadata.iterrows(), 
                         total=len(self.updated_metadata),
                         desc="Processing audio files"):
            
            audio_path = os.path.join(US8K_DIR, f"fold{row['fold']}", row['slice_file_name'])
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                continue
                
            # Load and preprocess audio
            y, sr = self.load_and_preprocess_audio(audio_path)
            if y is None:
                continue
                
            # Generate base mel spectrogram
            base_spec = self.create_mel_spectrogram(y, sr)
            if base_spec is None:
                continue
                
            # Create file identifier
            file_id = f"{row['fsID']}-{row['classID']}"
            fold = row['fold']
            
            # Save original spectrogram
            orig_spec_name = f"{file_id}-orig.png"
            orig_spec_path = os.path.join(SPEC_DIR, f"fold{fold}", orig_spec_name)
            self.save_spectrogram_image(base_spec, orig_spec_path)
            
            # Add to metadata
            spec_metadata.append({
                'spec_file_name': orig_spec_name,
                'orig_file_name': row['slice_file_name'],
                'fsID': row['fsID'],
                'fold': fold,
                'classID': row['classID'],
                'class': row['class'],
                'augmentation': 'original'
            })
            
            # Apply augmentations
            augmentations = []
            
            # 1. Time stretching
            for i, rate in enumerate([0.8, 1.2]):
                if i < AUGMENTATION_PER_CLASS['time_stretch']:
                    y_stretched = self.time_stretch(y, rate)
                    spec = self.create_mel_spectrogram(y_stretched, sr)
                    aug_name = f"{file_id}-ts{i}.png"
                    aug_path = os.path.join(SPEC_DIR, f"fold{fold}", aug_name)
                    self.save_spectrogram_image(spec, aug_path)
                    
                    augmentations.append({
                        'spec_file_name': aug_name,
                        'orig_file_name': row['slice_file_name'],
                        'fsID': row['fsID'],
                        'fold': fold,
                        'classID': row['classID'],
                        'class': row['class'],
                        'augmentation': f'timestretch_{rate}'
                    })
            
            # 2. Pitch shifting
            for i, n_steps in enumerate([-2, 2]):
                if i < AUGMENTATION_PER_CLASS['pitch_shift']:
                    y_shifted = self.pitch_shift(y, sr, n_steps)
                    spec = self.create_mel_spectrogram(y_shifted, sr)
                    aug_name = f"{file_id}-ps{i}.png"
                    aug_path = os.path.join(SPEC_DIR, f"fold{fold}", aug_name)
                    self.save_spectrogram_image(spec, aug_path)
                    
                    augmentations.append({
                        'spec_file_name': aug_name,
                        'orig_file_name': row['slice_file_name'],
                        'fsID': row['fsID'],
                        'fold': fold,
                        'classID': row['classID'],
                        'class': row['class'],
                        'augmentation': f'pitchshift_{n_steps}'
                    })
            
            # 3. SpecAugment
            for i in range(AUGMENTATION_PER_CLASS['spec_augment']):
                spec_augmented = self.spec_augment(base_spec)
                aug_name = f"{file_id}-sa{i}.png"
                aug_path = os.path.join(SPEC_DIR, f"fold{fold}", aug_name)
                self.save_spectrogram_image(spec_augmented, aug_path)
                
                augmentations.append({
                    'spec_file_name': aug_name,
                    'orig_file_name': row['slice_file_name'],
                    'fsID': row['fsID'],
                    'fold': fold,
                    'classID': row['classID'],
                    'class': row['class'],
                    'augmentation': f'specaugment_{i}'
                })
            
            # 4. PatchAugment
            for i in range(AUGMENTATION_PER_CLASS['patch_augment']):
                patch_augmented = self.patch_augment(base_spec)
                aug_name = f"{file_id}-pa{i}.png"
                aug_path = os.path.join(SPEC_DIR, f"fold{fold}", aug_name)
                self.save_spectrogram_image(patch_augmented, aug_path)
                
                augmentations.append({
                    'spec_file_name': aug_name,
                    'orig_file_name': row['slice_file_name'],
                    'fsID': row['fsID'],
                    'fold': fold,
                    'classID': row['classID'],
                    'class': row['class'],
                    'augmentation': f'patchaugment_{i}'
                })
            
            # 5. Noise addition
            for i, snr in enumerate([5, 15]):
                if i < AUGMENTATION_PER_CLASS['noise']:
                    y_noisy = self.add_noise(y, snr)
                    spec = self.create_mel_spectrogram(y_noisy, sr)
                    aug_name = f"{file_id}-n{i}.png"
                    aug_path = os.path.join(SPEC_DIR, f"fold{fold}", aug_name)
                    self.save_spectrogram_image(spec, aug_path)
                    
                    augmentations.append({
                        'spec_file_name': aug_name,
                        'orig_file_name': row['slice_file_name'],
                        'fsID': row['fsID'],
                        'fold': fold,
                        'classID': row['classID'],
                        'class': row['class'],
                        'augmentation': f'noise_{snr}dB'
                    })
            
            # Add augmentations to metadata
            spec_metadata.extend(augmentations)
        
        # Create and save spectrogram metadata
        self.spec_metadata = pd.DataFrame(spec_metadata)
        self.spec_metadata.to_csv(SPEC_CSV, index=False)
        print(f"Generated {len(self.spec_metadata)} spectrograms")
        print(f"Spectrogram metadata saved to {SPEC_CSV}")
    
    def balance_classes(self):
        """Balance classes across all folds using undersampling."""
        print("Balancing classes across folds...")
        
        # Get current class distribution
        class_dist = self.spec_metadata['class'].value_counts()
        print(f"Initial class distribution: {class_dist}")
        
        # Find minority and majority classes
        min_class_count = class_dist.min()
        majority_classes = class_dist[class_dist > min_class_count * (1/TARGET_RATIO - 1)].index
        
        # Calculate target count per class (based on minority class)
        target_count = int(min_class_count * (1/TARGET_RATIO - 1))
        print(f"Target count per majority class: {target_count}")
        
        # Create a copy of the metadata for balanced dataset
        balanced_metadata = []
        
        # Process each fold separately to maintain stratification
        for fold in range(1, 11):
            fold_data = self.spec_metadata[self.spec_metadata['fold'] == fold]
            
            # For each class in the fold
            for class_name in self.spec_metadata['class'].unique():
                class_data = fold_data[fold_data['class'] == class_name]
                
                # If majority class, undersample
                if class_name in majority_classes:
                    # Calculate per-fold target count
                    fold_target = min(len(class_data), 
                                     math.ceil(target_count / 10))
                    
                    # Prioritize original samples over augmentations for retention
                    orig_samples = class_data[class_data['augmentation'] == 'original']
                    aug_samples = class_data[class_data['augmentation'] != 'original']
                    
                    # Keep all originals if possible, otherwise sample
                    if len(orig_samples) <= fold_target:
                        keep_orig = orig_samples
                        aug_to_keep = min(fold_target - len(orig_samples), len(aug_samples))
                        keep_aug = aug_samples.sample(aug_to_keep) if aug_to_keep > 0 else pd.DataFrame()
                    else:
                        keep_orig = orig_samples.sample(fold_target)
                        keep_aug = pd.DataFrame()  # No need for augmentations
                    
                    # Combine samples to keep
                    keep_samples = pd.concat([keep_orig, keep_aug])
                    
                else:
                    # For minority classes, keep all samples
                    keep_samples = class_data
                
                # Add to balanced metadata
                balanced_metadata.append(keep_samples)
        
        # Combine all balanced data
        self.balanced_metadata = pd.concat(balanced_metadata, ignore_index=True)
        
        # Save balanced metadata
        balanced_csv = os.path.join(OUTPUT_DIR, 'spectrograms_balanced.csv')
        self.balanced_metadata.to_csv(balanced_csv, index=False)
        
        # Report final distribution
        final_dist = self.balanced_metadata['class'].value_counts()
        print(f"Final class distribution: {final_dist}")
        print(f"Balanced metadata saved to {balanced_csv}")
    
    def run(self):
        """Run the complete dataset extension process."""
        self.load_metadata()
        self.analyze_sirenet_data()
        self.integrate_new_files()
        self.generate_spectrograms()
        self.balance_classes()
        print("Dataset extension complete!")

if __name__ == "__main__":
    extender = UrbanSoundExtender()
    extender.run()

#!/usr/bin/env python
# coding: utf-8

# In[51]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.notebook import tqdm


# In[4]:


def preprocess_audio(file_path):
    # Load audio file with librosa
    y, sr = librosa.load(file_path, sr=None)
    # Convert to mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    mel_spec_db_norm = (mel_spec_db + 80) / 80

    return mel_spec_db_norm

# Example usage
file_path = 'archive/fold1/193394-3-0-7.wav'
mel_spectrogram = preprocess_audio(file_path)

# Plotting the Mel-Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram, sr=48000, x_axis='time', y_axis='mel')
plt.axis('off')  # No axis for image
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig("./sample.png", bbox_inches='tight', pad_inches=0)
plt.show()

plt.close()


# In[56]:


all_files = []
for dirname, _, filenames in os.walk('./archive/'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        all_files.append(file_path)


# In[57]:


len(all_files)


# In[54]:


for i, file_path in enumerate(tqdm(all_files)):
    if file_path[-3:] == "wav":
        mel_spectrogram = preprocess_audio(file_path)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram, sr=48000, x_axis='time', y_axis='mel')
        plt.axis('off')  # No axis for image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.tight_layout()
        file_name = file_path.split('/')[-1]
        save_dir = file_path.split('/')[-2]
        save_path = os.path.join('./archive/', save_dir, file_name[:-3] + 'png')
        print(save_path)
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(str(i) + ". Saved at " + save_path)
        
        os.remove(file_path)
        print(str(i) + "Deleted " + file_path)
        plt.close()


# In[58]:


import pandas as pd
pd.read_csv('./archive/UrbanSound8K.csv')

# change .wav to png and save it to that same csv file

df = pd.read_csv('./archive/UrbanSound8K.csv')
df.head()



# In[60]:


df['slice_file_name'] = df['slice_file_name'].apply(lambda x: x[:-3] + 'png')
df.head()


# In[61]:


df.to_csv('./archive/UrbanSound8K.csv', index=False)  


# In[ ]:





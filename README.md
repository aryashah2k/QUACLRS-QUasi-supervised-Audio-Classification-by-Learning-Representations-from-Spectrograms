# QUACLRS Audio Classification App

A Streamlit web application for audio classification using the QUACLRS (QUasi-supervised Audio Classification by Learning Representations from Spectrograms) model.

## Features

- **Upload Audio**: Analyze your own audio files
- **Record Audio**: Record and analyze audio directly from your microphone
- **Example Audio**: Test with example audio files
- **Explainable AI Visualizations**: See what the model "hears" using GradCAM and Integrated Gradients
- **Class Probability Display**: View confidence scores for all classes

## Classes Supported

The model can classify audio into the following categories:
- air_conditioner
- car_horn
- children_playing
- dog_bark
- drilling
- engine_idling
- gun_shot
- jackhammer
- ambulance
- firetruck
- police
- traffic
- street_music

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open your browser at http://localhost:8501
3. Upload or record an audio file and click "Classify Audio"

## Project Structure

- `app.py`: The main Streamlit application
- `models/model.py`: QUACLRS model implementation
- `fold_10_checkpoint.pth`: Pre-trained model weights
- `TestAudioFiles/`: Directory for example audio files (optional)

## Requirements

See `requirements.txt` for the full list of dependencies.

## Notes

- For best results, use audio files with a sample rate of 22050 Hz
- Supported audio formats: WAV, MP3, OGG
- The TestAudioFiles directory needs to be created and populated with sample audio files to use the "Example Audio" feature

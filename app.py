import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import soundfile as sf
from PIL import Image
import time
import os
import base64
from models.model import AudioClassifier
import tempfile
import logging
from auth import is_authenticated, logout_user
from db import get_user_credit_balance, deduct_credits, get_config, init_db

# Initialize database
init_db()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get app configuration
config = get_config()
DEFAULT_CREDITS = config["app"]["default_credits"]
CREDITS_PER_INFERENCE = config["app"]["credits_per_inference"]

# Set page configuration
st.set_page_config(
    page_title="QUACLRS Audio Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #4A90E2;
    }
    .stAudio {
        margin-top: 1rem;
    }
    .credits-display {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        text-align: center;
        border: 1px solid #eee;
    }
    .stButton button {
        border-radius: 5px;
    }
    .user-info {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .user-avatar {
        background-color: #4A90E2;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the QUACLRS model (cached to avoid reloading)"""
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 
                  'dog_bark', 'drilling', 'engine_idling', 
                  'gun_shot', 'jackhammer', 'ambulance', 'firetruck', 'police', 'traffic', 'street_music']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(
        checkpoint_path='fold_10_checkpoint.pth',
        class_names=class_names,
        device=device
    )
    return model

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def create_visualization(viz_data, class_names, audio, sr):
    """Create visualization figures from model output"""
    
    # For top prediction probabilities
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    top_indices = np.argsort(viz_data['probs'])[::-1][:5]  # Top 5 predictions
    bars = ax1.barh(
        [class_names[i] for i in top_indices],
        [viz_data['probs'][i] * 100 for i in top_indices],
        color='#4A90E2'
    )
    ax1.set_xlabel('Probability (%)')
    ax1.set_title('Top 5 Class Probabilities')
    # Add percentage labels to bars
    for bar in bars:
        width = bar.get_width()
        ax1.text(
            width + 0.5, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', 
            ha='left', 
            va='center'
        )
    fig1_base64 = plot_to_base64(fig1)
    
    # For visualization grid
    fig2 = plt.figure(figsize=(15, 10))
    
    # 1. Original spectrogram
    ax1 = fig2.add_subplot(2, 3, 1)
    ax1.imshow(np.array(viz_data['original_img']))
    ax1.set_title("Original Spectrogram")
    ax1.axis('off')
    
    # 2. GradCAM heatmap
    ax2 = fig2.add_subplot(2, 3, 2)
    ax2.imshow(viz_data['gradcam_heatmap'], cmap='jet')
    ax2.set_title(f"GradCAM: {class_names[viz_data['top_class']]}")
    ax2.axis('off')
    
    # 3. GradCAM overlay
    ax3 = fig2.add_subplot(2, 3, 3)
    ax3.imshow(viz_data['gradcam_overlay'])
    ax3.set_title("GradCAM Overlay")
    ax3.axis('off')
    
    # 4. Integrated Gradients heatmap
    ax4 = fig2.add_subplot(2, 3, 4)
    ax4.imshow(viz_data['ig_heatmap'], cmap='viridis')
    ax4.set_title("Integrated Gradients")
    ax4.axis('off')
    
    # 5. Frequency-Time Analysis
    ax5 = fig2.add_subplot(2, 3, 5)
    spec = ax5.imshow(viz_data['spectrogram_db'], aspect='auto', origin='lower', cmap='viridis')
    fig2.colorbar(spec, ax=ax5, format='%+2.0f dB')
    ax5.set_title("Spectrogram (dB)")
    ax5.set_xlabel('Time Frames')
    ax5.set_ylabel('Mel Frequency Bands')
    
    # 6. Waveform with actual audio data
    ax6 = fig2.add_subplot(2, 3, 6)
    time_axis = np.linspace(0, len(audio)/sr, len(audio))
    ax6.plot(time_axis, audio, color='#4A90E2')
    ax6.set_title("Audio Waveform")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Amplitude")
    
    plt.tight_layout()
    fig2_base64 = plot_to_base64(fig2)
    
    return fig1_base64, fig2_base64

def main():
    try:
        # Check authentication
        if not is_authenticated():
            st.switch_page("pages/login.py")
        
        # Title and description
        st.markdown("<h1 class='main-header'>QUACLRS Audio Classifier</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>QUasi-supervised Audio Classification by Learning Representations from Spectrograms</p>", unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            # User info
            st.markdown(
                f"""
                <div class="user-info">
                    <div class="user-avatar">{st.session_state.name[0].upper()}</div>
                    <div>
                        <strong>{st.session_state.name}</strong><br>
                        <small>{st.session_state.email}</small>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Credits display
            credit_balance = get_user_credit_balance(st.session_state.email)
            st.markdown(
                f"""
                <div class="credits-display">
                    <strong style="font-size: 1.2rem;">{credit_balance}</strong> Credits
                    <br><small>({CREDITS_PER_INFERENCE} credits per classification)</small>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Buy more credits button
            if st.button("💰 Buy More Credits", type="primary", use_container_width=True):
                st.switch_page("pages/credits.py")
            
            st.divider()
            
            # Logout button
            if st.button("Logout", type="secondary", use_container_width=True):
                logout_user()
                st.rerun()
        
        # Load model
        with st.spinner("Loading QUACLRS model..."):
            model = load_model()
            st.success("Model loaded successfully!")
        
        # Sidebar options
        st.sidebar.title("Options")
        
        # Audio input method
        input_method = st.sidebar.radio(
            "Select Input Method",
            ["Upload Audio File", "Record Live Audio", "Use Example Audio"]
        )
        
        # Visualization options
        show_xai = st.sidebar.checkbox("Show Explainable AI Visualizations", value=True)
        
        # Track the audio file we'll use for classification
        audio_file = None
        file_name = None
        
        if input_method == "Upload Audio File":
            # File uploader
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
            if uploaded_file is not None:
                audio_file = uploaded_file
                file_name = uploaded_file.name
        
        elif input_method == "Record Live Audio":
            st.info("Use the microphone to record audio for classification")
            
            # Use Streamlit's native audio input
            recorded_audio = st.audio_input("Record your audio")
            
            if recorded_audio:
                st.success("Audio recorded successfully!")
                # Display the recorded audio
                st.audio(recorded_audio)
                
                audio_file = recorded_audio
                file_name = "recorded_audio.wav"
        
        elif input_method == "Use Example Audio":
            example_audio_dir = "TestAudioFiles"
            if os.path.exists(example_audio_dir):
                example_files = [f for f in os.listdir(example_audio_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
                if example_files:
                    selected_example = st.sidebar.selectbox("Select Example Audio", example_files)
                    file_name = selected_example
                    audio_file = os.path.join(example_audio_dir, selected_example)
                else:
                    st.sidebar.warning(f"No audio examples found in {example_audio_dir}")
            else:
                st.sidebar.warning(f"Example directory {example_audio_dir} not found")
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button("✨ Classify Audio", key="classify", type="primary", use_container_width=True)
        
        if process_button:
            if audio_file is not None:
                # Check if user has enough credits
                credit_balance = get_user_credit_balance(st.session_state.email)
                if credit_balance < CREDITS_PER_INFERENCE:
                    st.error(f"You don't have enough credits. You need {CREDITS_PER_INFERENCE} credits for classification, but you only have {credit_balance}.")
                    st.info("Purchase more credits to continue using the service.")
                    if st.button("Buy Credits Now", type="primary"):
                        st.switch_page("pages/credits.py")
                else:
                    # Deduct credits
                    success, new_balance = deduct_credits(st.session_state.email, CREDITS_PER_INFERENCE)
                    if not success:
                        st.error("Error deducting credits. Please try again.")
                        return
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process uploaded file
                    status_text.text("Processing audio...")
                    progress_bar.progress(10)
                    
                    # Handle different input types
                    temp_audio_path = None
                    try:
                        if isinstance(audio_file, str):
                            # It's a path to an example file
                            audio_path = audio_file
                        else:
                            # It's an uploaded file or recorded audio
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                tmp_file.write(audio_file.getvalue())
                                temp_audio_path = tmp_file.name
                            audio_path = temp_audio_path
                        
                        progress_bar.progress(30)
                        status_text.text("Running model inference...")
                        
                        # Run model prediction
                        start_time = time.time()
                        probs, audio, sr, viz_data = model.predict(audio_path, verbose=False, visualize_xai=show_xai)
                        inference_time = time.time() - start_time
                        
                        progress_bar.progress(70)
                        status_text.text("Generating visualizations...")
                        
                        # Get top prediction
                        top_idx = np.argmax(probs)
                        top_class = model.class_names[top_idx]
                        top_prob = probs[top_idx] * 100
                        
                        progress_bar.progress(90)
                        
                        # Display results
                        st.markdown(f"### 🎵 Audio: {file_name}")
                        
                        # Display credit usage information
                        st.info(f"Used {CREDITS_PER_INFERENCE} credits. You have {new_balance} credits remaining.")
                        
                        # Display audio player
                        st.audio(audio_path)
                        
                        # Show prediction result
                        st.markdown(
                            f"""
                            <div class="prediction-box">
                                <h2>🔍 Prediction Result</h2>
                                <h3>The audio is classified as: <span style="color:#4A90E2; font-size:1.8rem">{top_class.replace('_', ' ').title()}</span></h3>
                                <p>Confidence: <b>{top_prob:.2f}%</b></p>
                                <p><small>Inference time: {inference_time:.2f} seconds</small></p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Create and display visualizations
                        if show_xai and viz_data:
                            st.markdown("### 📊 Prediction Probabilities")
                            
                            probs_base64, viz_base64 = create_visualization(viz_data, model.class_names, audio, sr)
                            
                            # Show probability chart
                            st.markdown(f'<img src="data:image/png;base64,{probs_base64}" style="width:100%">', unsafe_allow_html=True)
                            
                            st.markdown("### 🔬 Explainable AI Visualizations")
                            st.markdown(f'<img src="data:image/png;base64,{viz_base64}" style="width:100%">', unsafe_allow_html=True)
                            
                            # Additional explanation
                            st.markdown("""
                            **Visualization Explained:**
                            - **GradCAM**: Highlights the regions of the spectrogram that influenced the model's decision.
                            - **Integrated Gradients**: Shows feature attributions across the spectrogram.
                            - **Spectrogram**: Frequency-time representation of the audio signal.
                            - **Waveform**: Time-domain representation of the audio signal.
                            """)
                        
                        progress_bar.progress(100)
                        status_text.text("Completed!")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
                        
                    finally:
                        # Clean up temporary file
                        if temp_audio_path and os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
            
            else:
                st.warning("Please upload, record, or select an audio file first.")
        
        # About section
        with st.expander("ℹ️ About QUACLRS"):
            st.markdown("""
            **QUACLRS** (QUasi-supervised Audio Classification by Learning Representations from Spectrograms) is a 
            neural network model for audio classification. It uses a modified MobileNetV3 architecture that was 
            trained using self-supervised learning techniques on spectrograms.
            
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
            
            **How it works:**
            1. The audio is converted to a mel-spectrogram
            2. The spectrogram is fed to the model
            3. The model outputs class probabilities
            4. Explainable AI techniques (GradCAM, Integrated Gradients) show what the model "heard"
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
        logger.error(f"Exception: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()

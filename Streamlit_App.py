import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from scipy.io.wavfile import write
import warnings
import tempfile
import os

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# Title and description
st.title('🎤 Speech Emotion Recognition')
st.markdown("---")

# Try to load the model with compatibility settings
@st.cache_resource
def load_model():
    try:
        # First attempt: Load with safe_mode=False
        model = tf.keras.models.load_model('ltsm_best_weights.hdf5', safe_mode=False)
    except:
        try:
            # Second attempt: Load with custom_objects
            model = tf.keras.models.load_model(
                'ltsm_best_weights.hdf5',
                custom_objects={'LSTM': tf.keras.layers.LSTM},
                safe_mode=False
            )
        except:
            try:
                # Third attempt: Load with compile=False
                model = tf.keras.models.load_model(
                    'ltsm_best_weights.hdf5',
                    compile=False,
                    safe_mode=False
                )
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
    return model

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to extract audio features
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)  # Fixed sample rate
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Load model
with st.spinner('Loading model...'):
    model = load_model()

if model is None:
    st.error("Failed to load model. Please check if the model file exists.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Try to import mic recorder
try:
    from streamlit_mic_recorder import mic_recorder
    
    # Audio recording section
    st.subheader("🎙️ Record Audio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("Click the button to start/stop recording")
        
    with col2:
        st.markdown("Speak clearly for 3-5 seconds")
    
    # Audio recorder
    audio = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        key='recorder'
    )
    
    if audio:
        # The audio is returned as a tuple (bytes, sample_rate)
        audio_bytes = audio['bytes']
        sample_rate = audio['sample_rate']
        
        # Display audio player
        st.audio(audio_bytes, format="audio/wav")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Recognize emotion button
        if st.button('🔍 Recognize Emotion', type='primary'):
            with st.spinner('Analyzing emotion...'):
                try:
                    # Extract features
                    features = extract_features(tmp_path)
                    
                    if features is not None:
                        # Prepare features for model
                        features = np.expand_dims(features, axis=0)
                        features = np.expand_dims(features, axis=-1)
                        
                        # Make prediction
                        predictions = model.predict(features, verbose=0)[0]
                        
                        # Get results
                        predicted_idx = np.argmax(predictions)
                        predicted_emotion = emotion_labels[predicted_idx]
                        confidence = predictions[predicted_idx] * 100
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        # Show main prediction
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Emotion", predicted_emotion.upper())
                        with col2:
                            st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Show all probabilities
                        st.markdown("### Probability Distribution")
                        for emotion, prob in zip(emotion_labels, predictions):
                            st.progress(float(prob), text=f"{emotion}: {prob*100:.1f}%")
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error during emotion recognition: {str(e)}")
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        1. Click **'Start Recording'** to begin recording
        2. Speak clearly for **3-5 seconds**
        3. Click **'Stop Recording'** when done
        4. Click **'Recognize Emotion'** to analyze your speech
        5. View the predicted emotion and confidence score
        
        **Note:** Make sure your microphone is enabled and you're in a quiet environment for best results.
        """)
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        This app uses a deep learning model (LSTM) trained on speech data to recognize emotions in human voice.
        
        **Supported emotions:**
        - 😠 Angry
        - 🤢 Disgust
        - 😨 Fear
        - 😊 Happy
        - 😐 Neutral
        - 😢 Sad
        - 😲 Surprise
        """)

except ImportError as e:
    st.error(f"Error importing mic_recorder: {str(e)}")
    st.info("""
    ⚠️ Please install streamlit-mic-recorder:
    
    Add this to your requirements.txt:
    ```
    streamlit-mic-recorder==1.0.0
    ```
    """)

# Add debug info in an expander
with st.expander("🔧 Debug Info"):
    st.write("Python version:", os.sys.version)
    st.write("TensorFlow version:", tf.__version__)
    st.write("Librosa version:", librosa.__version__)
    st.write("NumPy version:", np.__version__)

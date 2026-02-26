import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import warnings
warnings.filterwarnings('ignore')

# Try to load the model with compatibility settings
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
        # Third attempt: Load with compile=False
        model = tf.keras.models.load_model(
            'ltsm_best_weights.hdf5',
            compile=False,
            safe_mode=False
        )

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Function to extract audio features from the recorded audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Function to record audio
def record_audio(duration=5, samplerate=44100):
    st.write(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return recording, samplerate

# Main function for creating the Streamlit app
def main():
    st.title('🎤 Speech Emotion Recognition')
    st.write('Record your voice and check the predicted emotion!')

    # Initialize session state for audio data
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'samplerate' not in st.session_state:
        st.session_state.samplerate = None

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('🎙️ Start Recording', key='start'):
            try:
                audio, sr = record_audio(duration=5)
                st.session_state.audio_data = audio
                st.session_state.samplerate = sr
                st.success("Recording completed!")
            except Exception as e:
                st.error(f"Error during recording: {str(e)}")

    with col2:
        if st.button('▶️ Play Recording', key='play'):
            if st.session_state.audio_data is not None:
                st.audio(st.session_state.audio_data.flatten(), 
                        format='audio/wav', 
                        sample_rate=st.session_state.samplerate)
            else:
                st.warning('No recording available. Please record first.')

    with col3:
        if st.button('🔍 Recognize Emotion', key='recognize'):
            if st.session_state.audio_data is None:
                st.warning('No recorded audio found. Please record your voice first.')
            else:
                with st.spinner('Analyzing emotion...'):
                    try:
                        # Save temporary audio file
                        audio_path = 'temp_recording.wav'
                        write(audio_path, 
                              int(st.session_state.samplerate), 
                              (st.session_state.audio_data * 32767).astype(np.int16))
                        
                        # Extract features and predict
                        features = extract_features(audio_path)
                        features = np.expand_dims(features, axis=0)
                        features = np.expand_dims(features, axis=2)  # Add channel dimension if needed
                        
                        predicted_probabilities = model.predict(features, verbose=0)[0]
                        predicted_emotion = emotion_labels[np.argmax(predicted_probabilities)]
                        confidence = np.max(predicted_probabilities) * 100
                        
                        # Display results
                        st.success(f'🎯 Predicted Emotion: **{predicted_emotion.upper()}**')
                        st.info(f'Confidence: {confidence:.2f}%')
                        
                        # Show probability distribution
                        prob_dict = {emotion: f"{prob*100:.1f}%" 
                                   for emotion, prob in zip(emotion_labels, predicted_probabilities)}
                        st.write("Probability Distribution:")
                        st.json(prob_dict)
                        
                    except Exception as e:
                        st.error(f"Error during emotion recognition: {str(e)}")

    # Add instructions
    with st.expander("📋 Instructions"):
        st.write("""
        1. Click 'Start Recording' and speak for 5 seconds
        2. Click 'Play Recording' to hear your recording (optional)
        3. Click 'Recognize Emotion' to analyze your speech
        4. The app will predict the emotion in your voice
        """)

if __name__ == '__main__':
    main()

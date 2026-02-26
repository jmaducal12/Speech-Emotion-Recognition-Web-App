import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from scipy.io.wavfile import write
import warnings
warnings.filterwarnings('ignore')

# Try to load the model with compatibility settings
try:
    model = tf.keras.models.load_model('ltsm_best_weights.hdf5', safe_mode=False)
except:
    try:
        model = tf.keras.models.load_model(
            'ltsm_best_weights.hdf5',
            custom_objects={'LSTM': tf.keras.layers.LSTM},
            safe_mode=False
        )
    except:
        model = tf.keras.models.load_model(
            'ltsm_best_weights.hdf5',
            compile=False,
            safe_mode=False
        )

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Function to extract audio features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Main function
def main():
    st.title('🎤 Speech Emotion Recognition')
    st.write('Record your voice and check the predicted emotion!')

    # Try to import audio recorder component
    try:
        from streamlit_audio_recorder import audio_recorder
        
        # Use the browser-based audio recorder
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#ff4b4b",
            neutral_color="#6c757d",
            icon_name="microphone",
            key="audio_recorder"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            # Save the recorded audio temporarily
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_bytes)
            
            if st.button('🔍 Recognize Emotion'):
                with st.spinner('Analyzing emotion...'):
                    try:
                        # Extract features and predict
                        features = extract_features("temp_audio.wav")
                        features = np.expand_dims(features, axis=0)
                        
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
    
    except ImportError:
        st.warning("""
        ⚠️ Audio recorder component not installed. 
        
        To use the app, you need to:
        1. Add `streamlit-audio-recorder` to your requirements.txt
        2. Or create a `packages.txt` file with `portaudio19-dev`
        
        Please update your requirements.txt file.
        """)

    # Instructions
    with st.expander("📋 Instructions"):
        st.write("""
        1. Click the microphone button to start recording
        2. Speak clearly for a few seconds
        3. Click the button again to stop recording
        4. Click 'Recognize Emotion' to analyze your speech
        """)

if __name__ == '__main__':
    main()

import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

# Load the pre-trained model
model = tf.keras.models.load_model('ltsm_best_weights.hdf5')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Function to extract audio features from the recorded audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Main function for creating the Streamlit app
def main():
    st.title('Speech Emotion Recognition')
    st.write('Record your voice and check the predicted emotion!')

    recording = False
    audio_frames = []
    audio_path = 'recorded_audio.wav'

    # Live interactive for recording voice
    if st.button('Start Recording'):
        recording = True

    if st.button('Stop Recording'):
        recording = False
        sd.stop()

    if recording:
        audio = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
        audio_frames.append(audio)

    if len(audio_frames) > 0:
        st.audio(np.concatenate(audio_frames), format='audio/wav')

    if st.button('Recognize Emotion'):
        if len(audio_frames) == 0:
            st.warning('No recorded audio found. Please record your voice first.')
        else:
            write(audio_path, np.concatenate(audio_frames), 44100)
            features = extract_features(audio_path)
            features = np.expand_dims(features, axis=0)
            predicted_probabilities = model.predict(features)[0]
            predicted_emotion = emotion_labels[np.argmax(predicted_probabilities)]
            st.success(f'Predicted Emotion: {predicted_emotion}')

if __name__ == '__main__':
    main()
    

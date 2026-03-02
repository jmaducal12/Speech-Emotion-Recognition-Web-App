import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import warnings
import tempfile
import os

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Speech Emotion Recognition")
st.markdown("---")


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "ltsm_best_weights.hdf5",
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully!")


# ---------------------------------------------------
# EMOTION LABELS
# ---------------------------------------------------
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ---------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------
def extract_features(audio_bytes):
    try:
        # Save bytes to temp wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name

        # Load audio safely
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)

        if y is None or len(y) == 0:
            st.error("Audio data could not be processed.")
            return None

        # Extract MFCC
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled

    except Exception as e:
        st.error(f"❌ Feature extraction error: {str(e)}")
        return None


# ---------------------------------------------------
# AUDIO INPUT (BUILT-IN STREAMLIT)
# ---------------------------------------------------
st.subheader("🎙️ Record Audio")

audio_file = st.audio_input("Click to record and speak for 3–5 seconds")

if audio_file is not None:

    st.audio(audio_file)

    if st.button("🔍 Recognize Emotion", type="primary"):

        with st.spinner("Analyzing emotion..."):

            audio_bytes = audio_file.read()

            if audio_bytes is None or len(audio_bytes) == 0:
                st.error("No audio detected. Please try again.")
            else:

                features = extract_features(audio_bytes)

                if features is not None:

                    # Prepare input
                    features = np.expand_dims(features, axis=0)
                    features = np.expand_dims(features, axis=-1)

                    # Predict
                    prediction = model.predict(features, verbose=0)[0]

                    predicted_idx = np.argmax(prediction)
                    predicted_emotion = emotion_labels[predicted_idx]
                    confidence = prediction[predicted_idx] * 100

                    st.markdown("---")
                    st.subheader("📊 Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Predicted Emotion", predicted_emotion.upper())

                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")

                    st.markdown("### Probability Distribution")

                    for emotion, prob in zip(emotion_labels, prediction):
                        st.progress(
                            float(prob),
                            text=f"{emotion}: {prob*100:.1f}%"
                        )


# ---------------------------------------------------
# INSTRUCTIONS
# ---------------------------------------------------
with st.expander("📋 Instructions"):
    st.markdown("""
    1. Click the recorder
    2. Speak clearly for 3–5 seconds
    3. Stop recording
    4. Click **Recognize Emotion**
    5. View prediction results
    """)


# ---------------------------------------------------
# DEBUG INFO
# ---------------------------------------------------
with st.expander("🔧 Debug Info"):
    st.write("Python version:", os.sys.version)
    st.write("TensorFlow version:", tf.__version__)
    st.write("Librosa version:", librosa.__version__)
    st.write("NumPy version:", np.__version__)

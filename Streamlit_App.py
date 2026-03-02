import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import warnings
import tempfile
import os

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Speech Emotion Recognition")
st.markdown("---")


# ---------------------------------------------------
# Load Model (Safe Version)
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


model = load_model()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully!")


# ---------------------------------------------------
# Emotion Labels
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
# Feature Extraction
# ---------------------------------------------------
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)

        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled

    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None


# ---------------------------------------------------
# Mic Recorder
# ---------------------------------------------------
try:
    from streamlit_mic_recorder import mic_recorder

    st.subheader("🎙️ Record Audio")

    audio = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        key="recorder"
    )

    if audio:

        audio_bytes = audio["bytes"]

        st.audio(audio_bytes, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        if st.button("🔍 Recognize Emotion", type="primary"):

            with st.spinner("Analyzing emotion..."):

                features = extract_features(tmp_path)

                if features is not None:

                    features = np.expand_dims(features, axis=0)
                    features = np.expand_dims(features, axis=-1)

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

        os.unlink(tmp_path)

except ImportError:
    st.error("⚠️ streamlit-mic-recorder not installed properly.")


# ---------------------------------------------------
# Debug Section
# ---------------------------------------------------
with st.expander("🔧 Debug Info"):
    st.write("Python:", os.sys.version)
    st.write("TensorFlow:", tf.__version__)
    st.write("Librosa:", librosa.__version__)
    st.write("NumPy:", np.__version__)

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
        st.error(f"❌ Feature extraction error: {e}")
        return None


# ---------------------------------------------------
# MICROPHONE RECORDER
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

        # Play recorded audio
        st.audio(audio_bytes, format="audio/wav")

        # Save to temp file (DO NOT DELETE manually)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        if st.button("🔍 Recognize Emotion", type="primary"):

            with st.spinner("Analyzing emotion..."):

                features = extract_features(tmp_path)

                if features is not None:

                    # Prepare input shape
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

except ImportError:
    st.error("⚠️ streamlit-mic-recorder not installed properly.")


# ---------------------------------------------------
# INSTRUCTIONS
# ---------------------------------------------------
with st.expander("📋 Instructions"):
    st.markdown("""
    1. Click **Start Recording**
    2. Speak clearly for 3–5 seconds
    3. Click **Stop Recording**
    4. Click **Recognize Emotion**
    5. View results and confidence score
    """)


# ---------------------------------------------------
# DEBUG SECTION
# ---------------------------------------------------
with st.expander("🔧 Debug Info"):
    st.write("Python version:", os.sys.version)
    st.write("TensorFlow version:", tf.__version__)
    st.write("Librosa version:", librosa.__version__)
    st.write("NumPy version:", np.__version__)

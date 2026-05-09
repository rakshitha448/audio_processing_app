import streamlit as st
import numpy as np
from scipy.io import wavfile
import tempfile

st.title("Audio Frequency Processor")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # Read audio
    rate, data = wavfile.read(temp_file.name)

    st.write("Sample Rate:", rate)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # FFT
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_data), 1/rate)

    # Frequency control
    low = st.slider("Low Frequency", 0, 20000, 300)
    high = st.slider("High Frequency", 0, 20000, 3000)
    gain = st.slider("Gain", 0.0, 3.0, 1.0)

    # Apply filter
    for i in range(len(freqs)):
        if low < abs(freqs[i]) < high:
            fft_data[i] *= gain

    # Inverse FFT
    new_audio = np.fft.ifft(fft_data).real

    # Normalize
    new_audio = np.int16(new_audio / np.max(np.abs(new_audio)) * 32767)

    # Save output
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(output_file.name, rate, new_audio)

    st.audio(output_file.name)

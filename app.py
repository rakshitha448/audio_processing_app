import streamlit as st
from pydub import AudioSegment
import imageio_ffmpeg
import tempfile
import numpy as np
from scipy.fft import fft, ifft

# Fix FFmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

st.title("🎵 Audio Processing App")

# Upload
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Load audio
    audio = AudioSegment.from_file(temp_path, format="wav")

    st.subheader("Original Audio")
    st.audio(uploaded_file)

    # Convert to numpy
    samples = np.array(audio.get_array_of_samples())

    # FFT
    fft_data = fft(samples)
    freqs = np.fft.fftfreq(len(fft_data), d=1/audio.frame_rate)

    # Controls
    target_freq = st.slider("Select Frequency (Hz)", 100, 5000, 1000)
    gain = st.slider("Gain (-10 to 10)", -10, 10, 0)

    # Modify frequency
    for i in range(len(freqs)):
        if abs(freqs[i] - target_freq) < 100:
            fft_data[i] *= (1 + gain)

    # Inverse FFT
    processed_samples = np.real(ifft(fft_data))

    # Convert back to audio
    processed_audio = audio._spawn(processed_samples.astype(np.int16).tobytes())

    # Save output
    output_path = temp_path + "_processed.wav"
    processed_audio.export(output_path, format="wav")

    st.subheader("Processed Audio")
    st.audio(output_path)

    # Download
    with open(output_path, "rb") as f:
        st.download_button(
            "Download Processed Audio",
            f,
            file_name="processed.wav"
        )

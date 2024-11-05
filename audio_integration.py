import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import io

st.title("Live Audio Input")

# Settings for audio input
duration = st.slider("Duration (seconds)", 1, 10, 5)  # Duration of audio to record
fs = 44100  # Sample rate

# Function to record audio
def record_audio(duration, fs):
    st.write("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    st.write("Recording complete")
    return audio

# Record audio when button is pressed
if st.button("Record Audio"):
    audio_data = record_audio(duration, fs)
    
    
    # Convert audio to 16-bit PCM for saving
    audio_data_16bit = np.int16(audio_data * 32767)
    
    # Save audio to a buffer as a valid WAV file
    wav_buffer = io.BytesIO()
    write(wav_buffer, fs, audio_data_16bit)
    wav_buffer.seek(0)

    # Option to download the audio as WAV
    st.write("Download audio file:")
    st.download_button("Download", data=wav_buffer, file_name="audio.wav", mime="audio/wav")

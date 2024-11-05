import streamlit as st
import sounddevice as sd
import numpy as np
import soundfile as sf
import io
import speech_recognition as sr

# Set audio parameters
samplerate = 44100  # Sample rate
duration = 10  # seconds

def record_audio():
    st.write("Recording... Please speak.")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished.")
    return audio_data

def audiorec_demo_app():
    if st.sidebar.button("Start Recording"):
        audio_data = record_audio()
        
        # Save audio data to a WAV file in memory
        wav_file = io.BytesIO()
        sf.write(wav_file, audio_data, samplerate, format='WAV')
        wav_file.seek(0)

        # Playback
        st.audio(wav_file, format='audio/wav')

        # Transcribe audio
        text = transcribe_audio(wav_file.getvalue())
        if text:
            st.write("Transcribed Text:")
            st.write(text)
        else:
            st.error("Could not transcribe audio.")

def transcribe_audio(audio_data):
    recognizer = sr.Recognizer()
    audio_file = io.BytesIO(audio_data)

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Speech could not be understood.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None

if __name__ == '__main__':
    audiorec_demo_app()

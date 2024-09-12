import streamlit as st
import yt_dlp
import whisper
import os
import subprocess
import glob
from openai import OpenAI
from tqdm import tqdm

# OpenAI API anahtarını çevresel değişkenlerden al
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)

@st.cache_resource
def load_whisper_model(model_name):
    return whisper.load_model(model_name)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        st.success("ffmpeg is installed and working.")
    except subprocess.CalledProcessError:
        st.error("Error: ffmpeg is not installed or not working properly.")
        return False
    return True

def download_youtube_audio(url, output_path="audio"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'progress_hooks': [progress_hook],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        audio_files = glob.glob(f"{output_path}*.mp3")
        if audio_files:
            audio_path = audio_files[0]
            st.success(f"Audio file successfully downloaded: {audio_path}")
            st.info(f"File size: {os.path.getsize(audio_path)} bytes")
            return audio_path
        else:
            st.error(f"Error: No audio file found with the pattern {output_path}*.mp3")
            return None
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None

def progress_hook(d):
    if d['status'] == 'downloading':
        total = d.get('total_bytes')
        downloaded = d.get('downloaded_bytes', 0)
        if total:
            progress = downloaded / total
            st.session_state.progress_bar.progress(progress)
            st.write(f"Downloaded: {downloaded}/{total} bytes ({progress:.2%})")
    elif d['status'] == 'finished':
        st.session_state.progress_bar.progress(1.0)
        st.success("Download completed!")

def transcribe_audio(audio_path, model_name, language):
    model = load_whisper_model(model_name)
    result = model.transcribe(audio_path, language=language)
    return result

def format_transcript(transcript):
    formatted_text = ""
    for segment in transcript['segments']:
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        formatted_text += f"[{start_time} - {end_time}] {segment['text']}\n"
    return formatted_text.strip()

def format_timestamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def summarize_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly skilled AI trained in language comprehension and summarization. Read the following text and summarize it into a concise abstract paragraph. Retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Avoid unnecessary details or tangential points."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in summarizing text: {str(e)}")
        return None

def process_video():
    if not check_ffmpeg():
        return

    st.session_state.progress_bar = st.progress(0)
    
    with st.spinner("Downloading YouTube video..."):
        audio_path = download_youtube_audio(st.session_state.youtube_url)

    if not audio_path:
        st.error("Error: Audio file not found. Download may have failed.")
        return

    with st.spinner("Transcribing audio..."):
        try:
            transcript = transcribe_audio(audio_path, st.session_state.whisper_model, st.session_state.language)
        except Exception as e:
            st.error(f"Error in transcribing audio: {str(e)}")
            return

    with st.spinner("Formatting transcript..."):
        st.session_state.formatted_transcript = format_transcript(transcript)

    st.session_state.processing_done = True

def main():
    st.title("YouTube Video Transcriber and Summarizer")

    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False

    if 'formatted_transcript' not in st.session_state:
        st.session_state.formatted_transcript = ""

    st.session_state.youtube_url = st.text_input("Enter the YouTube video URL:", key="url_input")
    
    st.session_state.language = st.selectbox("Select language", ["en", "tr"], format_func=lambda x: "English" if x == "en" else "Turkish")
    
    st.session_state.whisper_model = st.selectbox("Select Whisper model", ["tiny", "base", "small", "medium", "large"])
    
    if st.button("Process Video"):
        process_video()

    if st.session_state.processing_done:
        st.subheader("Transcript")
        st.text_area("Full Transcript", st.session_state.formatted_transcript, height=300)

        if st.button("Summarize Transcript"):
            with st.spinner("Summarizing transcript..."):
                summary = summarize_text(st.session_state.formatted_transcript)
            
            if summary:
                st.subheader("Summary")
                st.text_area("Transcript Summary", summary, height=200)
            else:
                st.error("Failed to generate summary. Please try again.")

if __name__ == "__main__":
    main()

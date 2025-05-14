import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import soundfile as sf
import io
import random
import openai

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="VibeVerse", page_icon="🎧")
st.title("🎧 VibeVerse – Psychedelic Music & Visual Generator")
st.markdown("Welcome to **VibeVerse** – where sound meets color, and AI speaks the vibe. 🌌")

# --- Load OpenAI API key securely ---
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.warning("🔐 OpenAI API key not found. Add it in `.streamlit/secrets.toml` or via Streamlit Cloud secrets.")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("🎵 Music Generator Settings")
duration = st.sidebar.slider("Duration (seconds)", 5, 60, 20)
freq = st.sidebar.slider("Base Frequency (Hz)", 100, 800, 440)
waveform = st.sidebar.selectbox("Waveform", ["sine", "sawtooth", "square", "noise"])

st.sidebar.header("🖼️ Visual Settings")
visual_style = st.sidebar.selectbox("Visual Style", ["Fractal", "Kaleido", "Warp"])
color_intensity = st.sidebar.slider("Color Intensity", 1, 10, 5)

# --- AUDIO GENERATION ---
def generate_tone(freq, duration, waveform):
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if waveform == "sine":
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
    elif waveform == "sawtooth":
        wave = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif waveform == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == "noise":
        wave = np.random.uniform(-1, 1, t.shape)
    return wave.astype(np.float32), sr

def generate_audio_buffer(wave, sr):
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, wave, sr, format='WAV')
    audio_buffer.seek(0)
    return audio_buffer

# --- VISUAL GENERATION ---
def generate_psychedelic_image(style):
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for i in range(512):
        for j in range(512):
            r = int((i * j) % 255)
            g = int((i ** 2 + j ** 2) % 255)
            b = int((np.sin(i * 0.1) * np.cos(j * 0.1) * 255) % 255)
            pixels[i, j] = (r, g, b)

    img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 3)))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color_intensity)

    if style == "Kaleido":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif style == "Warp":
        img = img.rotate(random.randint(10, 45))

    return img

# --- AI CAPTION GENERATION ---
def generate_caption(waveform, visual_style):
    prompt = f"""
    Write a short, poetic, psychedelic caption that fits a trippy visual inspired by {visual_style} style and {waveform} music.
    Make it cosmic, spiritual, and mysterious. Only 1 sentence.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=60
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return "✨ The AI mind is silent for now. (OpenAI error)"

# --- MAIN INTERACTION ---
if st.button("✨ Generate Psychedelic Music & Visual"):
    # Generate audio
    wave, sr = generate_tone(freq, duration, waveform)
    audio_buffer = generate_audio_buffer(wave, sr)
    st.audio(audio_buffer, format="audio/wav")

    # Generate visual
    img = generate_psychedelic_image(visual_style)
    st.image(img, caption="🌀 Psychedelic Visual", use_column_width=True)

    # Generate caption
    caption = generate_caption(waveform, visual_style)
    st.markdown(f"🌌 *{caption}*")

    # Show waveform snapshot
    fig, ax = plt.subplots()
    ax.plot(wave[:2000], alpha=0.6)
    ax.set_title("Audio Waveform Snapshot")
    ax.axis("off")
    st.pyplot(fig)

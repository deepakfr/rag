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

# --- Load OpenAI API Key ---
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.warning("🔐 OpenAI API key not found. Add it in `.streamlit/secrets.toml` or via Streamlit Cloud secrets.")
    st.stop()

# --- Text Prompt for AI Music Config ---
st.subheader("🧠 Describe Your Sound Vibe")
user_prompt = st.text_input(
    "Type what you want to hear:",
    placeholder="e.g. dark techno, dreamy ambient, alien jungle, etc."
)

# --- SIDEBAR CONTROLS (fallback) ---
st.sidebar.header("🎛️ Manual Controls (fallback if AI fails)")
default_duration = st.sidebar.slider("Duration (seconds)", 5, 60, 20)
default_freq = st.sidebar.slider("Base Frequency (Hz)", 100, 800, 440)
default_waveform = st.sidebar.selectbox("Waveform", ["sine", "sawtooth", "square", "noise"])

st.sidebar.header("🖼️ Visual Settings")
visual_style = st.sidebar.selectbox("Visual Style", ["Fractal", "Kaleido", "Warp"])
color_intensity = st.sidebar.slider("Color Intensity", 1, 10, 5)

# --- AI INTERPRETATION OF USER PROMPT ---
def interpret_prompt(prompt):
    system_prompt = """
You are a music AI. Based on the user's request, return an audio config as JSON with:
- waveform: one of "sine", "sawtooth", "square", or "noise"
- frequency: integer from 100 to 800 (Hz)
- duration: integer from 5 to 60 (seconds)
Example: {"waveform": "sawtooth", "frequency": 130, "duration": 30}
Do NOT explain anything. Only return the raw JSON object.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        content = response.choices[0].message['content']
        return eval(content)
    except Exception:
        return {
            "waveform": default_waveform,
            "frequency": default_freq,
            "duration": default_duration
        }

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
    except Exception:
        return "✨ The AI mind is silent for now. (OpenAI error)"

# --- MAIN GENERATION ---
if st.button("✨ Generate Vibe"):
    # Get audio config from AI or fallback
    if user_prompt.strip():
        config = interpret_prompt(user_prompt)
        waveform = config["waveform"]
        freq = config["frequency"]
        duration = config["duration"]
        st.markdown(f"🎶 *AI interpreted your request as:* `{waveform}` wave at `{freq} Hz` for `{duration}s`")
    else:
        waveform = default_waveform
        freq = default_freq
        duration = default_duration
        st.markdown("🛠️ *Using manual settings from the sidebar.*")

    # Generate audio
    wave, sr = generate_tone(freq, duration, waveform)
    audio_buffer = generate_audio_buffer(wave, sr)
    st.audio(audio_buffer, format="audio/wav")

    # Generate image
    img = generate_psychedelic_image(visual_style)
    st.image(img, caption="🌀 Psychedelic Visual", use_column_width=True)

    # Generate caption
    caption = generate_caption(waveform, visual_style)
    st.markdown(f"🌌 *{caption}*")

    # Display waveform
    fig, ax = plt.subplots()
    ax.plot(wave[:2000], alpha=0.6)
    ax.set_title("Audio Waveform Snapshot")
    ax.axis("off")
    st.pyplot(fig)

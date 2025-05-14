import streamlit as st
import random
import openai
from PIL import Image, ImageEnhance, ImageFilter
import io

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="VibeVerse", page_icon="🎧")
st.title("🎧 VibeVerse – AI Music Explorer")
st.markdown("Describe your vibe. Get real tracks. Feel the universe. 🌌")

# --- Load OpenAI API Key ---
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.warning("🔐 OpenAI API key not found. Add it in `.streamlit/secrets.toml` or via Streamlit Cloud secrets.")
    st.stop()

# --- Visual Generation ---
def generate_psychedelic_image():
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for i in range(512):
        for j in range(512):
            r = int((i * j) % 255)
            g = int((i ** 2 + j ** 2) % 255)
            b = int((random.uniform(0.5, 1.5) * (np.sin(i * 0.1) * np.cos(j * 0.1) * 255)) % 255)
            pixels[i, j] = (r, g, b)
    img = img.filter(ImageEnhance.Color(img).enhance(2).filter(ImageFilter.GaussianBlur(2)))
    return img

# --- AI Function to Suggest Tracks ---
def suggest_tracks(prompt):
    system_prompt = """
You're a music recommendation assistant. Given a mood or genre, return 5 real tracks as bullet points.
Each item must include:
- 🎵 Title
- 👤 Artist
- 🔗 YouTube link (only YouTube)
Format using Markdown, clean and simple.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=600
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return "⚠️ Something went wrong while fetching music suggestions."

# --- Search Input ---
st.subheader("🧠 Describe your music vibe")
vibe_input = st.text_input("Type a genre or mood (e.g. 'dark techno', 'psychedelic trance', 'lofi jungle'):")

# --- Generate on Button Click ---
if st.button("🔮 Find My Vibe"):
    if not vibe_input.strip():
        st.warning("Please describe a vibe to get music suggestions.")
    else:
        st.markdown("🎧 Searching tracks for your vibe...")
        
        # Show visual
        img = generate_psychedelic_image()
        st.image(img, caption="🌀 Your AI-Generated Visual Vibe", use_column_width=True)
        
        # Suggest tracks
        tracks = suggest_tracks(vibe_input)
        st.markdown("### 🎶 Recommended Tracks")
        st.markdown(tracks)

        # Optionally extract and embed first YouTube video
        import re
        urls = re.findall(r'(https?://[^\s]+)', tracks)
        for url in urls[:1]:  # only embed the first one
            if "youtube" in url:
                st.video(url)

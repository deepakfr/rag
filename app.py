import streamlit as st
import requests
import time

# --- CONFIG ---
st.set_page_config(page_title="VibeVerse - AI Music Generator", layout="centered")
st.title("🎧 VibeVerse – Generate Music with AI")
st.markdown("Enter a musical vibe, and let AI compose it for you using Meta’s MusicGen 🎶")

# --- Get Hugging Face API token ---
try:
    api_token = st.secrets["huggingface"]["api_token"]
except KeyError:
    st.error("🚨 Hugging Face API token missing. Add it to `.streamlit/secrets.toml`.")
    st.stop()

headers = {
    "Authorization": f"Bearer {api_token}"
}

API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-medium"


# --- Prompt input ---
prompt = st.text_input("🎼 Describe the music you want:", placeholder="e.g. Dark techno with ambient pads")

if st.button("🎵 Generate Music"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        st.markdown("⏳ Generating music… please wait 15–30 seconds...")

        # Send to MusicGen
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

        if response.status_code == 503:
            st.warning("⏳ Model is loading… retrying in a few seconds.")
            time.sleep(10)
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

        if response.ok:
            with open("musicgen_output.wav", "wb") as f:
                f.write(response.content)
            st.audio("musicgen_output.wav", format="audio/wav")
            st.success("✅ Music generated!")
        else:
            st.error(f"❌ Failed to generate music: {response.status_code}")

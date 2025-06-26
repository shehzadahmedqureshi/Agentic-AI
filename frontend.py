# Optional: Load environment variables if not using pipenv
# from dotenv import load_dotenv
# load_dotenv()

# === Streamlit User Interface for Conversational AI ===
import streamlit as st

st.set_page_config(page_title="Conversational AI Playground", layout="centered")
st.title("Conversational AI Playground")
st.write("Build and chat with your own AI agent!")

definition = st.text_area("Describe your agent's behavior:", height=70, placeholder="e.g., You are a creative assistant...")

GROQ_MODELS = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

provider_choice = st.radio("Choose Model Provider:", ("Groq",))

if provider_choice == "Groq":
    chosen_model = st.selectbox("Pick a Groq Model:", GROQ_MODELS)

enable_search = st.checkbox("Enable Web Search")

user_input = st.text_area("Type your message:", height=150, placeholder="What would you like to ask?")

API_ENDPOINT = "http://127.0.0.1:9999/chat"

if st.button("Send to Agent"):
    if user_input.strip():
        # === Send Request to Backend ===
        import requests

        payload = {
            "model": chosen_model,
            "provider": provider_choice,
            "prompt": definition,
            "messages": [user_input],
            "search_enabled": enable_search
        }

        response = requests.post(API_ENDPOINT, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent's Reply")
                st.markdown(f"**Response:** {response_data}")




import streamlit as st
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Muslim GPT", page_icon="🌙", layout="centered")
st.title("🌙 Muslim GPT")
st.caption("An AI assistant for Islamic guidance, history, and knowledge.")

# Get API Key
api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    st.warning("Please configure your GEMINI_API_KEY in the .env file later.")

# Initialize Gemini Client
@st.cache_resource
def get_client(key):
    if key:
        return genai.Client(api_key=key)
    return None

client = get_client(api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("Ask a question..."):
    # Display and save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        if client:
            # We try our primary model first, then fall back to an alternate free tier model if exhausted
            models_to_try = ['gemini-2.0-flash', 'gemini-2.5-flash']
            response_successful = False
            
            for model_name in models_to_try:
                try:
                    # Format full chat history for the Gemini API
                    formatted_contents = []
                    for msg in st.session_state.messages:
                        api_role = "user" if msg["role"] == "user" else "model"
                        formatted_contents.append(
                            types.Content(
                                role=api_role,
                                parts=[types.Part.from_text(text=msg["content"])]
                            )
                        )

                    # System Instructions
                    system_instruction = (
                        "You are Muslim GPT, a respectful, knowledgeable, and empathetic AI assistant "
                        "specializing in Islamic history, jurisprudence, Quranic studies, and daily guidance. "
                        "Always provide references from the Quran or authentic Hadith where applicable, "
                        "and maintain a polite and objective tone."
                    )
                    
                    # Call API
                    response = client.models.generate_content(
                        model=model_name,
                        contents=formatted_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction
                        )
                    )
                    
                    output_text = response.text
                    st.markdown(output_text)
                    
                    # Save assistant response to memory
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    response_successful = True
                    break # Break out of the loop since it worked!
                    
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a quota/rate limit error
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        if model_name != models_to_try[-1]:
                            st.info(f"Primary model ({model_name}) is busy. Trying backup free model...")
                            continue
                        else:
                            # If all models fail, capture the retry time if possible or give a clean warning
                            st.error("Free tier limit reached! Please wait roughly 30 seconds before typing your next message to reset your free quota.")
                    else:
                        st.error(f"Error checking {model_name}: {e}")
                        break
        else:
            st.info("The system is ready! Add your API key to start chatting with full memory.")
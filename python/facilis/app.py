import streamlit as st
from facilis import extract_api_spec

# Set page config
st.set_page_config(page_title="API Extractor", layout="wide")

# App title
st.title("Facilis")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_input = st.chat_input("Describe the API integration you need...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process API extraction
    response = extract_api_spec(user_input)

    # Display assistant response
    with st.chat_message("assistant"):
        if isinstance(response, dict) and "error" in response:
            st.markdown(f"**Error:** {response['error']}")
        else:
            st.markdown("### Extracted OpenAPI JSON:")
            st.code(response, language="json")

    st.session_state.messages.append({"role": "assistant", "content": response})

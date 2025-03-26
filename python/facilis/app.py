import streamlit as st
import json
import os
from facilis import extract_api_specs, production_agent

# ✅ Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Facilis API Extractor", layout="wide")

# ✅ Initialize session state correctly
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'production_info' not in st.session_state:
    try:
        response = production_agent()
        if isinstance(response, str):  # Ensure it's a JSON string before loading
            st.session_state['production_info'] = json.loads(response)
        elif isinstance(response, dict):  # Already a dictionary
            st.session_state['production_info'] = response
        else:
            raise ValueError("Invalid response format")
    except (json.JSONDecodeError, ValueError):
        st.error("⚠️ Failed to parse production info. Please try again.")
        st.session_state['production_info'] = {}

# 🏷️ **App Title & Instructions**
st.title("Facilis API Extractor")
st.write("Send API descriptions to extract structured API specifications.")

# 🏷️ **Show Production Info (if available)**
if st.session_state['production_info']:
    st.write("**Production Name:**", st.session_state['production_info'].get('production_name', 'N/A'))
    st.write("**Namespace:**", st.session_state['production_info'].get('namespace', 'N/A'))

# 🏷️ **Chat Interface**
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 🏷️ **User Input**
user_input = st.chat_input("Enter API description (one or multiple endpoints, one per line):")
if user_input:
    st.session_state['messages'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # ✅ Extract API specifications for each endpoint
    responses = []
    endpoints = user_input.strip().split("\n")
    for endpoint_description in endpoints:
        try:
            extracted_data = extract_api_specs(endpoint_description)  # Correct function name
            if isinstance(extracted_data, str):  
                extracted_data = json.loads(extracted_data)  # Ensure JSON format
            responses.append(extracted_data)
        except json.JSONDecodeError:
            st.error("⚠️ Error parsing extracted API specification.")
            responses.append({"error": "Invalid JSON format in extracted data."})
        except Exception as e:
            st.error(f"⚠️ Error extracting API specification: {e}")
            responses.append({"error": str(e)})
    
    # ✅ Combine into a single OpenAPI JSON
    openapi_combined = {
        "openapi": "3.0.0",
        "info": {
            "title": st.session_state['production_info'].get("production_name", "Unnamed Production"),
            "version": "1.0.0"
        },
        "paths": {}
    }
    
    for response in responses:
        if isinstance(response, dict) and "error" in response:
            response_text = f"❌ Error: {response['error']}"  
        else:
            response_text = "✅ Successfully extracted API specification."
            try:
                openapi_combined["paths"].update(response.get("paths", {}))
            except Exception as e:
                st.error(f"❌ Error processing API response: {e}")
                response_text = f"Error: {e}"
        
        st.session_state['messages'].append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    
    # 🏷️ **Show final OpenAPI JSON**
    with st.expander("✅ View OpenAPI JSON"):
        st.json(openapi_combined)

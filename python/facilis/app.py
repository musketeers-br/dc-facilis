import streamlit as st
import yaml
import json
from facilis import IrisI14yService, ProductionData, process_api_integration
from pathlib import Path

current_dir = Path(__file__).parent
image_path = current_dir / "logo.png"

st.markdown("""
    <style>
        /* Center the image container */
        .block-container {
            padding-top: 1rem;
        }
        
        /* Custom title styling */
        .custom-title {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1E1E1E;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Create three columns to center the image
col1, col2, col3 = st.columns([1, 2, 1])

# Use the middle column to display the image
with col2:
    st.image(str(image_path), width=215)  # Set exact width, height will maintain aspect ratio

if 'production_data' not in st.session_state:
    st.session_state.production_data = ProductionData()
    
iris_service = IrisI14yService()

st.subheader("Enter API Endpoints")
st.write("Enter one endpoint per line. Example:")
st.code("""
    GET api.example.com/users to retrieve all users
    POST api.example.com/users to create a new user
""")

endpoints_input = st.text_area("API Endpoints", height=200)

if st.button("Process and Integrate"):
    if endpoints_input:
        status_container = st.container()
        #with st.spinner("Processing endpoints and integrating with Iris..."):
        with status_container:
            st.subheader("Processing Endpoints and Integrating with Iris")
            try:
                result = process_api_integration(
                    endpoints_input,
                    st.session_state.production_data,
                    iris_service,
                    st_container=status_container
                )
                
                st.subheader("Processing Results")
                
                # Safely access review details
                review_details = result.get('review_details', {})
                review_status = review_details.get('is_valid', False)
                st.markdown(f"Review Status: {'✓ Approved' if review_status else '✗ Not Approved'}")
                
                # Safely access iris integration results
                iris_result = result.get('iris_integration', {})
                if isinstance(iris_result, dict):
                    status_color = "green" if iris_result.get('success', False) else "red"
                    message = iris_result.get('message', 'No message available')
                    st.markdown(f"Iris Integration: <span style='color:{status_color}'>{message}</span>", 
                                unsafe_allow_html=True)
                    
                    if 'iris_response' in iris_result:
                        with st.expander("Iris Integration Details"):
                            st.json(iris_result['iris_response'])
                else:
                    st.error("Invalid Iris integration result format")
                
                # Only show OpenAPI documentation if we have it
                openapi_doc = result.get('openapi_documentation')
                if openapi_doc:
                    st.subheader("OpenAPI Documentation")
                    tab1, tab2 = st.tabs(["YAML", "JSON"])
                    
                    with tab1:
                        st.code(yaml.dump(openapi_doc, sort_keys=False),
                                language='yaml')
                    with tab2:
                        st.json(openapi_doc)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="Download YAML",
                            data=yaml.dump(openapi_doc, sort_keys=False),
                            file_name="openapi_spec.yaml",
                            mime="text/yaml"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(openapi_doc, indent=2),
                            file_name="openapi_spec.json",
                            mime="application/json"
                        )
                    
                    with col3:
                        report_data = {
                            'review_details': review_details,
                            'iris_integration': iris_result
                        }
                        st.download_button(
                            label="Download Integration Report",
                            data=json.dumps(report_data, indent=2),
                            file_name="integration_report.json",
                            mime="application/json"
                        )
            
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
                st.error("Please check the logs for more details.")
    else:
        st.warning("Please enter at least one endpoint.")

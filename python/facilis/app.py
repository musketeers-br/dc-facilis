import streamlit as st
import yaml
import json
from facilis import IrisI14yService, ProductionData, process_api_integration
from pathlib import Path

current_dir = Path(__file__).parent
image_path = current_dir / "logo.png"

st.markdown("""
    <style>
        /* Center the image and make it full width */
        .full-width-image {
            width: 100%;
            margin-bottom: 1rem;
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
        
        /* Container for better spacing */
        .container {
            padding: 0;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    # Display the image
    st.image(str(image_path), use_container_width=True) 
    
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
        with st.spinner("Processing endpoints and integrating with Iris..."):
            try:
                result = process_api_integration(
                    endpoints_input,
                    st.session_state.production_data,
                    iris_service
                )
                
                st.subheader("Processing Results")
                
                review_status = result['review_details']['is_valid']
                st.markdown(f"Review Status: {'✓ Approved' if review_status else '✗ Not Approved'}")
                
                iris_result = result['iris_integration']
                status_color = "green" if iris_result['success'] else "red"
                st.markdown(f"Iris Integration: <span style='color:{status_color}'>{iris_result['message']}</span>", 
                            unsafe_allow_html=True)
                
                if 'iris_response' in iris_result:
                    with st.expander("Iris Integration Details"):
                        st.json(iris_result['iris_response'])
                
                st.subheader("OpenAPI Documentation")
                tab1, tab2 = st.tabs(["YAML", "JSON"])
                
                with tab1:
                    st.code(yaml.dump(result['openapi_documentation'], 
                                    sort_keys=False),
                            language='yaml')
                with tab2:
                    st.json(result['openapi_documentation'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="Download YAML",
                        data=yaml.dump(result['openapi_documentation'], 
                                        sort_keys=False),
                        file_name="openapi_spec.yaml",
                        mime="text/yaml"
                    )
                
                with col2:
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(result['openapi_documentation'], 
                                        indent=2),
                        file_name="openapi_spec.json",
                        mime="application/json"
                    )
                
                with col3:
                    st.download_button(
                        label="Download Integration Report",
                        data=json.dumps({
                            'review_details': result['review_details'],
                            'iris_integration': result['iris_integration']
                        }, indent=2),
                        file_name="integration_report.json",
                        mime="application/json"
                    )
            
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
    else:
        st.warning("Please enter at least one endpoint.")
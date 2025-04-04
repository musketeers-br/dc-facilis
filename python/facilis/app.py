import streamlit as st
import yaml
import json
import asyncio
from facilis import ProductionData, process_api_integration
from pathlib import Path

current_dir = Path(__file__).parent
image_path = current_dir / "logo.png"

def display_cls_files(cls_files_data):
    """Display the generated CLS files with download buttons"""
    if not cls_files_data:
        return
    
    st.header("Generated IRIS Classes")
    
    # Create tabs for viewing all files
    tabs = st.tabs(["All Classes", "Individual Classes"])
    
    with tabs[0]:
        # Create a combined view of all classes
        all_classes_content = "\n\n// ===================================\n\n".join(
            f"// {filename}\n{content}" 
            for filename, content in cls_files_data.items()
        )
        
        st.code(all_classes_content, language="objectscript")
        
        # Download button for all classes as a zip
        if all_classes_content:
            create_download_button(
                "Download All Classes (ZIP)",
                create_zip_from_cls_files(cls_files_data),
                "iris_classes.zip",
                "application/zip"
            )
    
    with tabs[1]:
        # Create an expander for each class file
        for filename, content in cls_files_data.items():
            with st.expander(f"📄 {filename}"):
                st.code(content, language="objectscript")
                create_download_button(
                    f"Download {filename}",
                    content,
                    filename,
                    "text/plain"
                )

def create_zip_from_cls_files(cls_files_data):
    """Create a zip file containing all CLS files"""
    import io
    import zipfile
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in cls_files_data.items():
            zip_file.writestr(filename, content)
    
    return zip_buffer.getvalue()

def create_download_button(label, data, filename, mime_type):
    """Create a download button for file data"""
    import base64
    
    if isinstance(data, str):
        data = data.encode()
    
    b64 = base64.b64encode(data).decode()
    
    download_button_str = f'''
        <a href="data:{mime_type};base64,{b64}" download="{filename}">
            <button style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 12px 30px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;">
                💾 {label}
            </button>
        </a>
    '''
    st.markdown(download_button_str, unsafe_allow_html=True)

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

async def main():
    if 'production_data' not in st.session_state:
        st.session_state.production_data = ProductionData()
        

    st.markdown("""
        This tool helps you convert natural language API descriptions into OpenAPI specifications 
        and integrate them with your production environment. Simply:
        
        1. **Describe your APIs** naturally - just write what you want them to do
        2. **Provide production details** - name and namespace for your deployment
        3. **Let the AI agents** handle the conversion and integration
        
        The tool will:
        - Extract API endpoints from your description
        - Generate proper OpenAPI specifications
        - Validate the endpoints and parameters
        - Create interoperability with your InterSystems IRIS environment
        """)

    # Initialize session state if needed
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    input_container = st.container()
    status_container = st.container()
    result_container = st.container()

    with input_container:
        endpoints_input = st.text_area("Prompt",
            height=150,
            help="Enter endpoint descriptions, one per line",
            placeholder="I want to retrieve all users from api.example.com/users using GET method and also create a new user POST api.example.com/users",
            key="endpoints_input"
        )
        
        if st.button("Process APIs"):
            if not endpoints_input:
                st.error("Please enter at least one API endpoint.")
            else:
                st.session_state.processing = True
    if st.session_state.processing:
        with status_container:
            st.subheader("Processing Endpoints and Integrating with Iris")
            try:
                result = await process_api_integration(
                    endpoints_input,
                    st.session_state.production_data,
                    st_container=status_container
                )
                
                st.subheader("Processing Results")
                
                # Safely access review details
                review_details = result.get('review_details', {})
                review_status = review_details.get('is_valid', False)
                st.markdown(f"Review Status: {'✓ Approved' if review_status else '✗ Not Approved'}")
                
                # Display OpenAPI documentation
                openapi_doc = result.get('openapi_documentation')
                if openapi_doc:
                    st.subheader("OpenAPI Documentation")
                    tab1, tab2 = st.tabs(["YAML", "JSON"])
                    
                    with tab1:
                        st.code(yaml.dump(openapi_doc, sort_keys=False),
                                language='yaml')
                    with tab2:
                        st.json(openapi_doc)
                    
                    col1, col2 = st.columns(2)
                    
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
                
                # Display generated CLS files
                cls_files = result.get('generated_files', {})
                if cls_files:
                    display_cls_files(cls_files)
                    st.success("✅ IRIS Classes generated successfully!")
                else:
                    st.warning("No IRIS classes were generated.")

            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
                st.error("Please check the logs for more details.")

if __name__ == "__main__":
    asyncio.run(main())
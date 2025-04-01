#!/usr/bin/env python
# coding: utf-8

import os
from crewai import Agent, Task, Crew
from textwrap import dedent
import json
import requests
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(f'logs/facilis_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('facilis')

class StreamlitCallback:
    def __init__(self, container: Optional[Any] = None):
        self.container = container or st
        self.current_agent = None
        self.logger = logging.getLogger('facilis.StreamlitCallback')
        self.user_inputs = {}

    def on_crew_start(self):
        self.logger.info("Starting CrewAI workflow")
        self.container.info("🚀 Starting API integration process...")

    def on_crew_end(self):
        self.logger.info("CrewAI workflow completed")
        self.container.success("✅ API integration process completed!")

    def on_agent_start(self, agent: Agent, task: Any):
        self.current_agent = agent
        # Handle both string and Task objects
        task_description = task.description if hasattr(task, 'description') else str(task)
        message = f"👤 Agent '{agent.role}' starting task: {task_description[:100]}..."
        self.logger.info(message)
        self.container.info(message)

    def on_agent_end(self, agent: Agent, task: Any):
        self.current_agent = None
        # Handle both string and Task objects
        task_description = task.description if hasattr(task, 'description') else str(task)
        message = f"✨ Agent '{agent.role}' completed task: {task_description[:100]}"
        self.logger.info(message)
        self.container.success(message)

    def on_timeout_error(self, retry_count: int, max_retries: int):
        """New method to notify user about timeout and retries"""
        if retry_count < max_retries:
            message = f"""⏳ Request timeout occurred (attempt {retry_count}/{max_retries})
            The server is taking longer than expected to respond.
            Automatically retrying in a moment..."""
            self.logger.warning(message)
            self.container.warning(message)
        else:
            message = """🔄 The server is currently experiencing high latency.
            
            Suggestions:
            1. Wait a few moments and try again
            2. Check your network connection
            3. Verify the server status
            4. If the problem persists, contact support
            
            Click the 'Generate' button to try again."""
            self.logger.error(message)
            self.container.error(message)

    def on_agent_error(self, agent: Any, task: Any, error: Exception):
        agent_role = agent.role if hasattr(agent, 'role') else str(agent)
        task_description = task.description if hasattr(task, 'description') else str(task)
        
        # Check if it's a timeout error
        if isinstance(error, IrisIntegrationError) and "timeout" in str(error).lower():
            self.on_timeout_error(3, 3)  # Assuming max retries is 3
            return
            
        # Handle other IRIS-related tasks
        if 'iris' in str(task_description).lower():
            self.on_iris_generation_complete(False, str(error))
        
        message = f"❌ Error in agent '{agent_role}' during task: {task_description[:100]}\nError: {str(error)}"
        self.logger.error(message)
        self.container.error(message)

    def on_iris_generation_start(self):
        """New method to notify when IRIS generation begins"""
        message = """🔄 Generating InterSystems IRIS interoperability components:
        - Creating Business Service
        - Setting up Business Process
        - Configuring Business Operation
        - Establishing Message Routes"""
        self.logger.info("Starting IRIS interoperability generation")
        self.container.info(message)

    def on_iris_generation_complete(self, status: bool, details: str = None):
        """New method to notify when IRIS generation completes"""
        if status:
            message = "✅ Successfully generated InterSystems IRIS interoperability components!"
            self.container.success(message)
        else:
            message = f"❌ Failed to generate InterSystems IRIS components: {details or 'Unknown error'}"
            self.container.error(message)
        self.logger.info(message)

    def request_user_input(self, field: str, field_type: str = "text", options: List = None, required: bool = False) -> Any:
        """
        Request input from the user through Streamlit interface
        """
        self.logger.info(f"Requesting user input for: {field}")
        
        # Create a stable key for the input field
        input_key = f"input_{field.lower().replace(' ', '_')}"
        
        self.container.write(f"📝 Please provide the following information:")
        label = f"Enter {field}:" if not required else f"Enter {field} (required):"
        
        user_input = None
        
        if field_type == "text":
            user_input = self.container.text_input(label, key=input_key)
        elif field_type == "select" and options:
            user_input = self.container.selectbox(
                label,
                options,
                key=input_key,
                help=f"Select a {field} from the list"
            )
        elif field_type == "json":
            user_input = self.container.text_area(
                label,
                height=150,
                key=input_key,
                help="Enter a valid JSON object"
            )
        
        if user_input:
            self.user_inputs[field] = user_input
            return user_input
        
        if required:
            self.container.error(f"{field} is required. Please provide a value.")
            st.stop()  # Stop execution until input is provided
        
        return None

    def get_production_details(self) -> Optional[Dict[str, str]]:
        """
        Get required production details from user
        """
        # Try to get production name
        production_name = self.request_user_input("production name", required=True)
        if not production_name:
            return None
            
        # Try to get namespace
        namespace = self.request_user_input("namespace", required=True)
        if not namespace:
            return None

        return {
            "production_name": production_name,
            "namespace": namespace
        }


    def get_http_method(self) -> str:
        """
        Get HTTP method from user with validation
        """
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        return self.request_user_input(
            "HTTP Method",
            field_type="select",
            options=valid_methods
        )

    def get_json_model(self, endpoint: str) -> Dict:
        """
        Get JSON model from user with validation
        """
        while True:
            json_str = self.request_user_input(
                f"JSON model for {endpoint}",
                field_type="json"
            )
            if not json_str:
                return None
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                self.container.error("Invalid JSON format. Please try again.")

class ProductionData:
    def __init__(self):
        self.logger = logging.getLogger('facilis.ProductionData')
        self.productions = {}  # Store productions and their endpoints
        
    def add_production(self, name: str, namespace: str):
        self.logger.info(f"Adding production: {name} in namespace: {namespace}")
        if name not in self.productions:
            self.productions[name] = {
                'namespace': namespace,
                'created_at': datetime.now().isoformat(),
                'endpoints': []
            }
            
    def add_endpoint(self, production_name: str, endpoint_data: Dict):
        self.logger.info(f"Adding endpoint to production: {production_name}")
        if production_name in self.productions:
            self.productions[production_name]['endpoints'].append(endpoint_data)
            
    def production_exists(self, name: str) -> bool:
        return name in self.productions

class OpenAPITransformer:
    @staticmethod
    def create_openapi_base(info: Dict) -> Dict:
        return {
            "openapi": "3.0.0",
            "info": {
                "title": info.get("production_name", "API Documentation"),
                "version": "1.0.0",
                "description": f"API documentation for {info.get('production_name')} in {info.get('namespace')}"
            },
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {}
            }
        }

    @staticmethod
    def create_path_item(endpoint_spec: Dict) -> Dict:
        method = endpoint_spec["HTTP_Method"].lower()
        path_item = {
            method: {
                "summary": endpoint_spec.get("description", ""),
                "parameters": [],
                "responses": {
                    "200": {
                        "description": "Successful response"
                    }
                }
            }
        }

        if endpoint_spec.get("params"):
            for param in endpoint_spec["params"]:
                path_item[method]["parameters"].append({
                    "name": param,
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"}
                })

        if method in ["post", "put", "patch"] and endpoint_spec.get("json_model"):
            path_item[method]["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": endpoint_spec["json_model"]
                    }
                }
            }

        return path_item

class IrisIntegrationError(Exception):
    """Custom exception for Iris integration errors"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error

class IrisI14yService:
    def __init__(self):
        self.logger = logging.getLogger('facilis.IrisI14yService')
        self.base_url = os.getenv("FACILIS_URL", "http://dc-facilis-iris-1:52773") 
        self.headers = {
            "Content-Type": "application/json"
        }
        self.timeout = int(os.getenv("IRIS_TIMEOUT", "504")) 
        self.max_retries = int(os.getenv("IRIS_MAX_RETRIES", "3")) 
        self.logger.info("IrisI14yService initialized")

    def get_namespaces(self) -> List[str]:
        """
        Get available namespaces from Iris
        """
        self.logger.info("Fetching namespaces from Iris")
        try:
            response = requests.get(
                f"{self.base_url}/facilis/api/namespaces",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to fetch namespaces: {str(e)}"
            self.logger.error(error_msg)
            return []

    def send_to_iris(self, payload: Dict) -> Dict:
        """
        Send payload to Iris generate endpoint with timeout handling and retry logic
        """
        self.logger.info("Sending payload to Iris generate endpoint")
        if isinstance(payload, str):
            try:
                json.loads(payload)  
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        
        retry_count = 0
        last_error = None

        while retry_count < self.max_retries:
            try:
                # Add more detailed logging
                self.logger.info(f"Attempt {retry_count + 1}/{self.max_retries}: Sending request to {self.base_url}/facilis/api/generate")
                
                response = requests.post(
                    f"{self.base_url}/facilis/api/generate",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout / 1000  # Convert ms to seconds
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as e:
                retry_count += 1
                last_error = e
                error_msg = f"Timeout occurred (attempt {retry_count}/{self.max_retries})"
                self.logger.warning(error_msg)
                
                if retry_count < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s...
                    wait_time = 2 ** (retry_count - 1)
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                continue

            except requests.exceptions.RequestException as e:
                error_msg = f"Failed to send to Iris: {str(e)}"
                self.logger.error(error_msg)
                raise IrisIntegrationError(error_msg)

        error_msg = f"Failed to send to Iris after {self.max_retries} attempts due to timeout"
        self.logger.error(error_msg)
        raise IrisIntegrationError(error_msg, last_error)

class APIAgents:
    def __init__(self, llm, iris_service: IrisI14yService):
        self.logger = logging.getLogger('facilis.APIAgents')
        self.llm = llm
        self.iris_service = iris_service
        self.logger.info("APIAgents initialized")

    def create_production_agent(self) -> Agent:
        self.logger.info("Creating Production Manager agent")
        return Agent(
            role='Production Manager',
            goal='Manage production environments and namespaces',
            backstory=dedent("""
                You are responsible for managing production environments and their namespaces.
                You interact with users to gather production details and validate their existence.
            """),
            allow_delegation=False,
            llm=self.llm
        )

    def create_interaction_agent(self) -> Agent:
        return Agent(
            role='User Interaction Specialist',
            goal='Interact with users to obtain missing API specification fields',
            backstory=dedent("""
                You are a specialist in user interaction, responsible for identifying
                and requesting missing information in API specifications.
            """),
            allow_delegation=False,
            llm=self.llm
        )

    def create_validation_agent(self) -> Agent:
        return Agent(
            role='API Validator',
            goal='Validate API specifications for correctness and consistency',
            backstory=dedent("""
                You are an expert in API validation, ensuring all specifications
                meet the required standards and format.
            """),
            allow_delegation=False,
            llm=self.llm
        )

    def create_extraction_agent(self) -> Agent:
        return Agent(
            role='API Specification Extractor',
            goal='Extract API specifications from natural language descriptions',
            backstory=dedent("""
                You are specialized in interpreting natural language descriptions
                and extracting structured API specifications.
            """),
            allow_delegation=True,
            llm=self.llm
        )

    def create_transformation_agent(self) -> Agent:
        return Agent(
            role='OpenAPI Transformation Specialist',
            goal='Convert API specifications into OpenAPI documentation',
            backstory=dedent("""
                You are an expert in OpenAPI specifications and documentation.
                Your role is to transform validated API details into accurate
                and comprehensive OpenAPI 3.0 documentation.
            """),
            allow_delegation=False,
            llm=self.llm
        )

    def create_reviewer_agent(self) -> Agent:
        return Agent(
            role='OpenAPI Documentation Reviewer',
            goal='Ensure OpenAPI documentation compliance and quality',
            backstory=dedent("""
                You are the final authority on OpenAPI documentation quality and compliance.
                With extensive experience in OpenAPI 3.0 specifications, you meticulously
                review documentation for accuracy, completeness, and adherence to standards.
            """),
            allow_delegation=True,
            llm=self.llm
        )

    def create_iris_i14y_agent(self) -> Agent:
        return Agent(
            role='Iris I14y Integration Specialist',
            goal='Integrate API specifications with Iris I14y service',
            backstory=dedent("""
                You are responsible for ensuring smooth integration between the API
                documentation system and the Iris I14y service. You handle the
                communication with Iris, validate responses, and ensure successful
                integration of API specifications.
            """),
            allow_delegation=False,
            llm=self.llm
        )

class APISpecificationCrew:
    def __init__(self, llm, production_data: ProductionData, iris_service: IrisI14yService, callback=None):
        self.callback = callback
        self.current_agent = None 
        api_agents = APIAgents(llm, iris_service)
        self.production_agent = api_agents.create_production_agent()
        self.interaction_agent = api_agents.create_interaction_agent()
        self.validation_agent = api_agents.create_validation_agent()
        self.extraction_agent = api_agents.create_extraction_agent()
        self.transformation_agent = api_agents.create_transformation_agent()
        self.reviewer_agent = api_agents.create_reviewer_agent()
        self.iris_i14y_agent = api_agents.create_iris_i14y_agent()
        self.production_data = production_data
        self.iris_service = iris_service

    def get_production_details(self) -> Task:
        if self.callback:
            self.callback.on_agent_start(self.production_agent, "Getting production details")
        return Task(
            description=dedent("""
                Interact with the user to obtain:
                1. Production name
                2. Namespace
                
                Check if the production exists in the system.
                If it doesn't exist, confirm if a new production should be created.
                
                Return results in JSON format:
                {
                    "production_name": string,
                    "namespace": string,
                    "exists": boolean,
                    "create_new": boolean
                }
            """),
            expected_output="""A JSON object containing production details including name, namespace, existence status, and creation flag""",
            agent=self.production_agent
        )

    def handle_missing_fields(self, missing_fields: List[str], endpoint_info: Dict) -> Task:
        description = dedent(f"""
            The following fields are missing for endpoint {endpoint_info.get('endpoint', 'unknown')}:
            {', '.join(missing_fields)}
            
            Current endpoint info: {json.dumps(endpoint_info, indent=2)}
            
            For each missing field:
            1. If HTTP_Method: Must be one of GET, POST, PUT, DELETE, PATCH
            2. If json_model: Required for POST/PUT/PATCH methods
            3. If params: List of parameter names
            4. If production_name: Name of the production environment
            5. If namespace: Namespace for the production
            
            Return the collected information in JSON format.
        """)

        if self.callback:
            updated_fields = {}
            
            for field in missing_fields:
                if field == "HTTP_Method":
                    method = self.callback.get_http_method()
                    if method:
                        updated_fields[field] = method
                
                elif field == "json_model":
                    if endpoint_info.get("HTTP_Method", "").upper() in ["POST", "PUT", "PATCH"]:
                        json_model = self.callback.get_json_model(endpoint_info.get("endpoint", ""))
                        if json_model:
                            updated_fields[field] = json_model
                
                elif field == "production_name":
                    name = self.callback.request_user_input("production name")
                    if name:
                        updated_fields[field] = name

                elif field == "namespace":
                    try:
                        namespaces = self.iris_service.get_namespaces()
                        if namespaces:
                            namespace = self.callback.request_user_input(
                                "namespace",
                                field_type="select",
                                options=namespaces,
                                required=True
                            )
                            if namespace:
                                updated_fields[field] = namespace
                        else:
                            namespace = self.callback.request_user_input("namespace")

                    except Exception as e:
                        self.callback.container.error(f"Failed to fetch namespaces: {str(e)}")
                        st.stop()

                else:
                    value = self.callback.request_user_input(field)
                    if value:
                        updated_fields[field] = value
            
            return Task(
                description=f"Process collected user inputs: {json.dumps(updated_fields, indent=2)}",
                expected_output="A JSON object containing the updated field values",
                agent=self.interaction_agent
            )
        
        return Task(
            description=description,
            expected_output="A JSON object containing the updated field values",
            agent=self.interaction_agent
        )

    def validate_api_spec(self, extracted_data: Dict) -> Task:
        return Task(
            description=dedent(f"""
                Validate the following API specification:
                {json.dumps(extracted_data, indent=2)}
                
                Check for:
                1. Valid host format
                2. Endpoint starts with '/'
                3. Valid HTTP method (GET, POST, PUT, DELETE, PATCH)
                4. Valid port number (if provided)
                5. JSON model presence for POST/PUT/PATCH/DELETE methods
                
                Return validation results in JSON format.
            """),
            expected_output="""A JSON object containing validation results with any errors or confirmation of validity""",
            agent=self.validation_agent
        )

    def extract_api_specs(self, descriptions: List[str]) -> Task:
        return Task(
            description=dedent(f"""
                Extract API specifications from the following descriptions:
                {json.dumps(descriptions, indent=2)}
                
                For each description, extract:
                - host (required)
                - endpoint (required)
                - HTTP_Method (required)
                - params (optional)
                - port (if available)
                - json_model (for POST/PUT/PATCH/DELETE)
                - authentication (if applicable)
                
                Mark any missing required fields as 'missing'.
                Return results in JSON format as an array of specifications.
            """),
            expected_output="""A JSON array containing extracted API specifications with all required and optional fields""",
            agent=self.extraction_agent
        )

    def transform_to_openapi(self, validated_endpoints: List[Dict], production_info: Dict) -> Task:
        return Task(
            description=dedent(f"""
                Transform the following validated API specifications into OpenAPI 3.0 documentation:
                
                Production Information:
                {json.dumps(production_info, indent=2)}
                
                Validated Endpoints:
                {json.dumps(validated_endpoints, indent=2)}
                
                Requirements:
                1. Generate complete OpenAPI 3.0 specification
                2. Include proper request/response schemas
                3. Document all parameters and request bodies
                4. Include authentication if specified
                5. Ensure proper path formatting
                
                Return the OpenAPI specification in both JSON and YAML formats.
            """),
            expected_output="""A JSON object containing the complete OpenAPI 3.0 specification with all endpoints and schemas""",
            agent=self.transformation_agent
        )

    def review_openapi_spec(self, openapi_spec: Dict) -> Task:
        return Task(
            description=dedent(f"""
                Review the following OpenAPI specification for compliance and quality:
                
                {json.dumps(openapi_spec, indent=2)}
                
                Review Checklist:
                1. OpenAPI 3.0 Compliance
                - Verify correct version specification
                - Check required root elements
                - Validate schema structure
                
                2. Completeness
                - All endpoints properly documented
                - Parameters fully specified
                - Request/response schemas defined
                - Security schemes properly configured
                
                3. Quality Checks
                - Consistent naming conventions
                - Clear descriptions
                - Proper use of data types
                - Meaningful response codes
                
                4. Best Practices
                - Proper tag usage
                - Consistent parameter naming
                - Appropriate security definitions
                
                You must return a JSON object with the following structure:
                {{
                    "is_valid": boolean,
                    "approved_spec": object (the reviewed and possibly corrected OpenAPI spec),
                    "issues": [array of strings describing any issues found],
                    "recommendations": [array of improvement suggestions]
                }}
            """),
            expected_output="""A JSON object containing: is_valid (boolean), approved_spec (object), issues (array), and recommendations (array)""",
            agent=self.reviewer_agent
        )

    def send_to_iris(self, openapi_spec: Dict, production_info: Dict, review_result: Dict) -> Task:
        return Task(
            description=dedent(f"""
                Send the approved OpenAPI specification to Iris I14y service:

                Production Information:
                - Name: {production_info['production_name']}
                - Namespace: {production_info['namespace']}
                - Is New: {production_info.get('create_new', False)}

                Review Status:
                - Approved: {review_result['is_valid']}
                
                Return the integration result in JSON format.
            """),
            expected_output="""A JSON object containing the integration result with Iris I14y service, including success status and response details""",
            agent=self.iris_i14y_agent
        )

def get_facilis_llm():
    """Returns the appropriate chat model based on AI_ENGINE selection."""
    logger = logging.getLogger('facilis.get_facilis_llm')
    
    ai_engine = os.getenv("AI_ENGINE")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME")
    
    logger.info(f"Initializing LLM with engine: {ai_engine}")

    if ai_engine == "openai":
        from openai import OpenAI
        logger.info(f"Using OpenAI with model: {model_name}")
        os.environ["OPENAI_API_KEY"] = api_key
        return model_name  # Return just the model name

    if ai_engine in ["azureopenai", "azure_openai"]:
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        if not azure_endpoint or not azure_deployment_name:
            logger.error("Azure OpenAI configuration missing")
            raise ValueError("Azure OpenAI requires AZURE_ENDPOINT and AZURE_DEPLOYMENT_NAME in .env")
        logger.info(f"Using Azure OpenAI with deployment: {azure_deployment_name}")
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        return azure_deployment_name  # Return the deployment name for Azure

    if ai_engine in ["anthropic", "claude"]:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        return model_name

    if ai_engine == "gemini":
        os.environ["GOOGLE_API_KEY"] = api_key
        return model_name
    
    if ai_engine == "ollama":
        return model_name
    
    logger.error(f"Invalid AI engine selected: {ai_engine}")
    return None

def extract_json_from_markdown(markdown_text: str) -> str:
    """
    Extracts JSON content from markdown-formatted string.
    Handles cases where JSON is wrapped in ```json or ``` code blocks.
    """
    logger = logging.getLogger('facilis.extract_json_from_markdown')
    try:
        # Convert the CrewOutput to string if it isn't already
        text = str(markdown_text)
        
        # If the text is already a valid JSON string, return it
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        if '```json' in text:
            # Split by ```json and take the content after it
            parts = text.split('```json')
            if len(parts) > 1:
                # Split by ``` to get the content between the code block
                json_text = parts[1].split('```')[0]
                return json_text.strip()
        elif '```' in text:
            # Split by ``` and take the content between code blocks
            parts = text.split('```')
            if len(parts) > 1:
                potential_json = parts[1].strip()
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    pass

        # If no code blocks, try to find JSON between curly braces
        start = text.find('{')
        end = text.rindex('}') + 1
        
        if start != -1 and end > start:
            json_str = text[start:end]
            # Validate that this is valid JSON
            json.loads(json_str)
            return json_str
            
        # If we couldn't find valid JSON, try to create a simple JSON object
        # from the text itself
        return json.dumps({"response": text})

    except Exception as e:
        logger.error(f"Failed to extract JSON from markdown: {e}")
        logger.debug(f"Original text: {text}")
        # Return a simple JSON object with the original text
        return json.dumps({"error": "Failed to parse response", "raw_response": text})


def process_api_integration(user_input: str, production_data: ProductionData, iris_service: IrisI14yService, st_container=None) -> Dict:
    logger = logging.getLogger('facilis.process_api_integration')
    logger.info("Starting API integration process")

    # Create the callback
    callback = StreamlitCallback(st_container)
    callback.on_crew_start()
    
    try:
        llm = get_facilis_llm()
        if llm is None:
            logger.error("Invalid AI_ENGINE selection")
            raise ValueError("Invalid AI_ENGINE selection")

        logger.info("Creating APISpecificationCrew")
        crew = APISpecificationCrew(llm, production_data, iris_service, callback)

        # Process endpoints
        logger.info("Processing endpoints")
        endpoints = user_input.split('\n')
        
        logger.info("Getting production details")
        api_crew = Crew(
            agents=[
                crew.production_agent,
                crew.extraction_agent,
                crew.validation_agent,
                crew.interaction_agent,
                crew.transformation_agent,
                crew.reviewer_agent,
                crew.iris_i14y_agent
            ],
            tasks=[crew.get_production_details()],
            verbose=True
        )

        production_result = api_crew.kickoff()
        production_info = json.loads(extract_json_from_markdown(production_result))

        if not isinstance(production_info, dict):
            raise ValueError("Invalid production info format")

        if not production_info.get('production_name') or not production_info.get('namespace'):
            logger.info("Waiting for production details...")
            st.stop()

        # Add production to production data if it doesn't exist
        if not production_data.production_exists(production_info['production_name']):
            production_data.add_production(
                production_info['production_name'],
                production_info['namespace']
            ) 
        production_info = callback.get_production_details()

        logger.info("Extracting API specifications")
        api_crew = Crew(
            agents=[crew.extraction_agent],
            tasks=[crew.extract_api_specs(endpoints)],
            verbose=True
        )

        extracted_json = extract_json_from_markdown(api_crew.kickoff())
        extracted_results = json.loads(extracted_json)
        
        if not isinstance(extracted_results, list):
            extracted_results = [extracted_results] if extracted_results else []

        final_endpoints = []
        for endpoint_spec in extracted_results:
            if not isinstance(endpoint_spec, dict):
                logger.warning(f"Skipping invalid endpoint spec: {endpoint_spec}")
                continue
                
            logger.info(f"Processing endpoint: {endpoint_spec.get('endpoint', 'unknown')}")
            missing_fields = [k for k, v in endpoint_spec.items() if v == 'missing']
            if missing_fields:
                logger.info(f"Handling missing fields: {missing_fields}")
                api_crew = Crew(
                    agents=[crew.interaction_agent],
                    tasks=[crew.handle_missing_fields(missing_fields, endpoint_spec)],
                    verbose=True
                )

                updated_json = extract_json_from_markdown(api_crew.kickoff())
                try:
                    updated_spec = json.loads(updated_json)
                    if isinstance(updated_spec, dict):
                        endpoint_spec.update(updated_spec)
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.error(f"Failed to update endpoint spec: {e}")

            logger.info("Validating endpoint specification")
            api_crew = Crew(
                agents=[crew.validation_agent],
                tasks=[crew.validate_api_spec(endpoint_spec)],
                verbose=True
            )

            validation_json = extract_json_from_markdown(api_crew.kickoff())
            validation_result = json.loads(validation_json)
            
            if isinstance(validation_result, dict) and 'error' not in validation_result:
                logger.info("Endpoint validation successful")
                final_endpoints.append(endpoint_spec)
                crew.production_data.add_endpoint(production_info['production_name'], endpoint_spec)
            else:
                logger.warning(f"Endpoint validation failed: {validation_result}")

        logger.info("Transforming to OpenAPI specification")
        api_crew = Crew(
            agents=[crew.transformation_agent],
            tasks=[crew.transform_to_openapi(final_endpoints, production_info)],
            verbose=True
        )
        openapi_result = json.loads(extract_json_from_markdown(api_crew.kickoff()))

        logger.info("Reviewing OpenAPI documentation")
        api_crew = Crew(
            agents=[crew.reviewer_agent],
            tasks=[crew.review_openapi_spec(openapi_result)],
            verbose=True
        )
        review_json = extract_json_from_markdown(api_crew.kickoff())
        review_result = json.loads(review_json)
        
        # Add validation and default values
        if not isinstance(review_result, dict):
            logger.warning("Invalid review result format, using default structure")
            review_result = {
                "is_valid": False,
                "approved_spec": openapi_result,
                "issues": ["Invalid review result format"],
                "recommendations": []
            }
        
        # Ensure required keys exist
        review_result.setdefault("is_valid", False)
        review_result.setdefault("approved_spec", openapi_result)
        review_result.setdefault("issues", [])
        review_result.setdefault("recommendations", [])

        if review_result["is_valid"]:
            logger.info("OpenAPI specification approved, sending to Iris")
            api_crew = Crew(
                agents=[crew.iris_i14y_agent],
                tasks=[crew.send_to_iris(review_result["approved_spec"], production_info, review_result)],
                verbose=True
            )
            iris_payload = {
                "production_name": production_info['production_name'],
                "namespace": production_info['namespace'],
                "openapi_spec": review_result["approved_spec"]
            }
            
            iris_result = iris_service.send_to_iris(iris_payload)
        else:
            logger.warning("OpenAPI specification not approved for integration")
            iris_result = {
                "success": False,
                "message": "OpenAPI specification not approved for integration",
                "timestamp": datetime.now().isoformat(),
                "issues": review_result.get("issues", [])
            }

    except Exception as e:
        callback.on_agent_error(
            agent=callback.current_agent if callback.current_agent else "Unknown",
            task="Current task",
            error=e
        )
        logger.error(f"Error in process_api_integration: {str(e)}")
        raise
    finally:
        callback.on_crew_end()

    logger.info("API integration process completed")
    return {
        'production_name': production_info['production_name'],
        'namespace': production_info['namespace'],
        'endpoints': final_endpoints,
        'openapi_documentation': review_result['approved_spec'],
        'review_details': review_result,
        'iris_integration': iris_result
    }

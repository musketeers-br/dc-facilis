#!/usr/bin/env python
# coding: utf-8

import os
from crewai import Agent, Task, Crew
from textwrap import dedent
import json
import requests
from typing import List, Dict
import logging
from datetime import datetime
from dotenv import load_dotenv

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


class IrisIntegrationError(Exception):
    """Custom exception for Iris integration errors"""
    pass

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

class IrisI14yService:
    def __init__(self):
        self.logger = logging.getLogger('facilis.IrisI14yService')
        self.base_url = os.getenv("IRIS_BASE_URL", "xxx")
        auth = ""
        if (os.getenv("IRIS_USERNAME") and os.getenv("IRIS_PASSWORD")):
            auth = f"{os.getenv('IRIS_USERNAME')}:{os.getenv('IRIS_PASSWORD')}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth}" if auth else None
        }
        self.logger.info("IrisI14yService initialized")

    def send_to_iris(self, payload: Dict) -> Dict:
        self.logger.info("Sending payload to Iris")
        try:
            response = requests.post(
                f"{self.base_url}/facilis",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise IrisIntegrationError(f"Failed to send to Iris: {str(e)}")

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
    def __init__(self, llm, production_data: ProductionData, iris_service: IrisI14yService):
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
        return Task(
            description=dedent(f"""
                Request the following missing fields from the user for endpoint {endpoint_info.get('endpoint', 'unknown')}:
                {', '.join(missing_fields)}
                
                For POST/PUT/PATCH/DELETE methods, ensure JSON model is provided.
                Current endpoint info: {json.dumps(endpoint_info, indent=2)}
            """),
            expected_output="""A JSON object containing the updated field values for the endpoint specification""",
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
                2. Completeness
                3. Quality Checks
                4. Best Practices
                
                Return a detailed review result in JSON format.
            """),
            expected_output="""A JSON object containing the review results, including validation status and any issues found""",
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
        
        # Find the first occurrence of '{'
        start = text.find('{')
        # Find the last occurrence of '}'
        end = text.rindex('}') + 1
        
        if start == -1 or end == 0:
            logger.warning("No JSON object found in markdown text")
            return text
        
        # Extract everything between the first { and last }
        json_str = text[start:end]
        
        # Validate that this is valid JSON
        json.loads(json_str)  # This will raise JSONDecodeError if invalid
        
        return json_str
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from markdown: {e}")
        raise
    except ValueError as e:
        logger.error(f"Failed to extract JSON from markdown: {e}")
        raise

def process_api_integration(user_input: str, production_data: ProductionData, iris_service: IrisI14yService) -> Dict:
    logger = logging.getLogger('facilis.process_api_integration')
    logger.info("Starting API integration process")

    llm = get_facilis_llm()
    if llm is None:
        logger.error("Invalid AI_ENGINE selection")
        raise ValueError("Invalid AI_ENGINE selection")

    logger.info("Creating APISpecificationCrew")
    crew = APISpecificationCrew(llm, production_data, iris_service)
    
    # Process endpoints
    logger.info("Processing endpoints")
    endpoints = user_input.split('\n')
    
    # Get production details
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
    
    if production_info['create_new']:
        logger.info(f"Creating new production: {production_info['production_name']}")
        crew.production_data.add_production(
            production_info['production_name'],
            production_info['namespace']
        )

    # Extract API specs
    logger.info("Extracting API specifications")
    api_crew = Crew(
        agents=[crew.extraction_agent],
        tasks=[crew.extract_api_specs(endpoints)],
        verbose=True
    )
    extracted_results = json.loads(extract_json_from_markdown(api_crew.kickoff()))
    
    final_endpoints = []
    for endpoint_spec in extracted_results:
        logger.info(f"Processing endpoint: {endpoint_spec.get('endpoint', 'unknown')}")
        missing_fields = [k for k, v in endpoint_spec.items() if v == 'missing']
        if missing_fields:
            logger.info(f"Handling missing fields: {missing_fields}")
            api_crew = Crew(
                agents=[crew.interaction_agent],
                tasks=[crew.handle_missing_fields(missing_fields, endpoint_spec)],
                verbose=True
            )
            updated_spec = json.loads(extract_json_from_markdown(api_crew.kickoff()))
            endpoint_spec.update(updated_spec)
        
        logger.info("Validating endpoint specification")
        api_crew = Crew(
            agents=[crew.validation_agent],
            tasks=[crew.validate_api_spec(endpoint_spec)],
            verbose=True
        )
        validation_result = json.loads(extract_json_from_markdown(api_crew.kickoff()))
        
        if 'error' not in validation_result:
            logger.info("Endpoint validation successful")
            final_endpoints.append(endpoint_spec)
            crew.production_data.add_endpoint(production_info['production_name'], endpoint_spec)
        else:
            logger.warning(f"Endpoint validation failed: {validation_result.get('error')}")

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
    review_result = json.loads(extract_json_from_markdown(api_crew.kickoff()))

    if review_result['is_valid']:
        logger.info("OpenAPI specification approved, sending to Iris")
        api_crew = Crew(
            agents=[crew.iris_i14y_agent],
            tasks=[crew.send_to_iris(openapi_result, production_info, review_result)],
            verbose=True
        )
        iris_result = json.loads(extract_json_from_markdown(api_crew.kickoff()))
    else:
        logger.warning("OpenAPI specification not approved for integration")
        iris_result = {
            "success": False,
            "message": "OpenAPI specification not approved for integration",
            "timestamp": datetime.now().isoformat()
        }

    if review_result['is_valid']:
        logger.info("OpenAPI specification approved, sending to Iris")
        api_crew = Crew(
            agents=[crew.iris_i14y_agent],
            tasks=[crew.send_to_iris(openapi_result, production_info, review_result)],
            verbose=True
        )
        iris_result = json.loads(str(api_crew.kickoff()))
    else:
        logger.warning("OpenAPI specification not approved for integration")
        iris_result = {
            "success": False,
            "message": "OpenAPI specification not approved for integration",
            "timestamp": datetime.now().isoformat()
        }

    logger.info("API integration process completed")
    return {
        'production_name': production_info['production_name'],
        'namespace': production_info['namespace'],
        'endpoints': final_endpoints,
        'openapi_documentation': review_result['approved_spec'],
        'review_details': review_result,
        'iris_integration': iris_result
    }

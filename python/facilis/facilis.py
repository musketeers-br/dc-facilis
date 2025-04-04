#!/usr/bin/env python
# coding: utf-8

import os
import json
import requests
import time
import logging
import aiohttp
import asyncio

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from textwrap import dedent
from typing import List, Dict, Any, Optional, Type, Union
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

OUTPUT_DIR = "/home/irisowner/dev/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        # Define message templates
        self.agent_messages = {
            "production": "Production Agent",
            "extraction": "Extraction Agent",
            "validation": "Validation Agent",
            "interaction": "Interaction Agent",
            "transformation": "Transformation Agent",
            "reviewer": "Reviewer Agent",
            "analyzer": "Analyzer Agent",
            "bs_generator": "Business Service Generator",
            "bo_generator": "Business Operation Generator",
            "exporter": "Exporter Agent",
            "collector": "Collector Agent"
        }

        self.completion_messages = {
            "reviewer": "✅ API specification review completed",
            "analyzer": "✅ OpenAPI structure analysis completed",
            "bs_generator": "✅ Business Service components generated",
            "bo_generator": "✅ Business Operation components generated",
            "exporter": "✅ Classes exported successfully",
            "collector": "✅ Generated files collected"
        }

    def format_agent_message(self, agent_role: str, task_description: str) -> str:
        """Format the agent start message"""
        default_message = "Agent {} starting task".format(agent_role)
        agent_msg = self.agent_messages.get(agent_role, default_message)
        return "👤 {}: {}...".format(agent_msg, task_description[:100])

    def format_completion_message(self, agent_role: str, task_description: str) -> str:
        """Format the agent completion message"""
        default_message = "Agent {} completed task".format(agent_role)
        completion_msg = self.completion_messages.get(agent_role, default_message)
        return "{}: {}".format(completion_msg, task_description[:100])

    def on_agent_start(self, agent: Agent, task: Any):
        self.current_agent = agent
        task_description = task.description if hasattr(task, 'description') else str(task)
        message = self.format_agent_message(agent.role, task_description)
        self.logger.info(message)
        self.container.info(message)

    def on_agent_end(self, agent: Agent, task: Any):
        self.current_agent = None
        task_description = task.description if hasattr(task, 'description') else str(task)
        message = self.format_completion_message(agent.role, task_description)
        self.logger.info(message)
        self.container.success(message)

    def on_crew_start(self):
        self.logger.info("Starting CrewAI workflow")
        self.container.info("🚀 Starting API integration process...")

    def on_crew_end(self):
        self.logger.info("CrewAI workflow completed")
        self.container.success("✅ API integration process completed!")

    def on_agent_error(self, agent: Any, task: Any, error: Exception):
        agent_role = agent.role if hasattr(agent, 'role') else str(agent)
        task_description = task.description if hasattr(task, 'description') else str(task)
        
        message = "❌ Error in agent '{}' during task: {}\nError: {}".format(
            agent_role,
            task_description[:100],
            str(error)
        )
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
        self.timeout = int(os.getenv("IRIS_TIMEOUT", "504"))  # in milliseconds
        self.max_retries = int(os.getenv("IRIS_MAX_RETRIES", "3"))
        self.logger.info("IrisI14yService initialized")

    async def send_to_iris_async(self, payload: Dict) -> Dict:
        """
        Send payload to Iris generate endpoint asynchronously
        """
        self.logger.info("Sending payload to Iris generate endpoint")
        if isinstance(payload, str):
            try:
                json.loads(payload)  
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        
        retry_count = 0
        last_error = None

        # Create timeout for aiohttp
        timeout = aiohttp.ClientTimeout(total=self.timeout / 1000)  # Convert ms to seconds

        while retry_count < self.max_retries:
            try:
                self.logger.info(f"Attempt {retry_count + 1}/{self.max_retries}: Sending request to {self.base_url}/facilis/api/generate")
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.base_url}/facilis/api/generate",
                        json=payload,
                        headers=self.headers
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        response.raise_for_status()

            except asyncio.TimeoutError as e:
                retry_count += 1
                last_error = e
                error_msg = f"Timeout occurred (attempt {retry_count}/{self.max_retries})"
                self.logger.warning(error_msg)
                
                if retry_count < self.max_retries:
                    wait_time = 2 ** (retry_count - 1)
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                continue

            except aiohttp.ClientError as e:
                error_msg = f"Failed to send to Iris: {str(e)}"
                self.logger.error(error_msg)
                raise IrisIntegrationError(error_msg)

        error_msg = f"Failed to send to Iris after {self.max_retries} attempts due to timeout"
        self.logger.error(error_msg)
        raise IrisIntegrationError(error_msg, last_error)

    # Keep the synchronous method for backward compatibility
    def send_to_iris(self, payload: Dict) -> Dict:
        """
        Synchronous wrapper for send_to_iris_async
        """
        return asyncio.run(self.send_to_iris_async(payload))

    async def get_namespaces_async(self) -> List[str]:
        """
        Get available namespaces from Iris asynchronously
        """
        self.logger.info("Fetching namespaces from Iris")
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout / 1000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{self.base_url}/facilis/api/namespaces",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    response.raise_for_status()
        except Exception as e:
            error_msg = f"Failed to fetch namespaces: {str(e)}"
            self.logger.error(error_msg)
            return []

    # Keep the synchronous method for backward compatibility
    def get_namespaces(self) -> List[str]:
        """
        Synchronous wrapper for get_namespaces_async
        """
        return asyncio.run(self.get_namespaces_async())


class IRISClassWriter:
    """Tool for generating InterSystems IRIS interoperability class files"""
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, output_dir=OUTPUT_DIR):
        if not IRISClassWriter._initialized:
            self.output_dir = os.path.abspath(output_dir)
            self.generated_classes = {}
            self._ensure_output_directory()
            IRISClassWriter._initialized = True

    def validate_classes(self, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate the generated IRIS classes
        
        Args:
            class_names (Optional[List[str]]): Optional list of specific class names to validate
            
        Returns:
            dict: Validation results for each class
        """
        validation_results = {}
        
        classes_to_validate = (
            class_names if class_names 
            else list(self.generated_classes.keys())
        )
        
        for class_name in classes_to_validate:
            if class_name not in self.generated_classes:
                validation_results[class_name] = {
                    "valid": False,
                    "issues": ["Class not found in generated classes"]
                }
                continue
                
            class_content = self.generated_classes[class_name]
            issues = []
            
            # Basic validation checks
            if not "Class " in class_content:
                issues.append("Missing Class declaration")
            
            if not "Extends " in class_content:
                issues.append("Missing Extends keyword")
            
            # Add more validation checks as needed
            
            validation_results[class_name] = {
                "valid": len(issues) == 0,
                "issues": issues
            }
        
        return validation_results

    def _ensure_output_directory(self):
        """Ensures the output directory exists and is writable"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            # Test if directory is writable
            test_file = os.path.join(self.output_dir, '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (IOError, OSError) as e:
                raise PermissionError(f"Output directory {self.output_dir} is not writable: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory {self.output_dir}: {str(e)}")
    
    def write_production_class(self, production_name, components):
        """
        Generate an IRIS Production class
        
        Args:
            production_name (str): Name of the production
            components (list): List of components to include in the production
            
        Returns:
            str: The generated class content
        """
        class_name = self._sanitize_class_name(production_name)
        
        class_content = f"""Class {class_name} Extends Ens.Production
        {{

        XData ProductionDefinition
        {{
        <Production Name="{class_name}" LogGeneralTraceEvents="false">
        """
            
        # Add components
        for component in components:
            comp_type = component.get("type", "")
            comp_name = component.get("name", "")
            comp_class = component.get("class", "")
            
            if comp_type and comp_name and comp_class:
                class_content += f"""  <{comp_type} Name="{comp_name}" Class="{comp_class}">
        </{comp_type}>
        """
                
                class_content += """</Production>
        }}

        }
        """
        
        self.generated_classes[class_name] = class_content
        return class_content
    
    def write_business_operation(self, operation_name, endpoint_info):
        """
        Generate an IRIS Business Operation class
        
        Args:
            operation_name (str): Name of the business operation
            endpoint_info (dict): Information about the API endpoint
            
        Returns:
            str: The generated class content
        """
        class_name = self._sanitize_class_name(f"bo{operation_name}")
        
        method = endpoint_info.get("method", "GET")
        path = endpoint_info.get("path", "/")
        
        class_content = f"""Class {class_name} Extends Ens.BusinessOperation
        {{

        Parameter ADAPTER = "EnsLib.HTTP.OutboundAdapter";

        Property Adapter As EnsLib.HTTP.OutboundAdapter;

        Parameter INVOCATION = "Queue";

        Method {operation_name}(pRequest As ms{operation_name}, Output pResponse As Ens.Response) As %Status
        {{
            Set tSC = $$$OK
            Try {{
                // Prepare HTTP request
                Set tHttpRequest = ##class(%Net.HttpRequest).%New()
                Set tHttpRequest.ContentType = "application/json"
                
                // Set request path and method
                Set tPath = "{path}"
                Set tMethod = "{method}"
                
                // Convert request message to JSON
                // [Additional logic for request preparation]
                
                // Send the HTTP request
                Set tSC = ..Adapter.SendFormDataArray(.tHttpResponse, tMethod, tPath, tHttpRequest)
                
                // Process response
                If $$$ISOK(tSC) {{
                    // Create response object
                    Set pResponse = ##class(Ens.Response).%New()
                    // Process HTTP response
                }}
            }}
            Catch ex {{
                Set tSC = ex.AsStatus()
            }}
            
            Return tSC
        }}

        XData MessageMap
        {{
        <MapItems>
        <MapItem MessageType="ms{operation_name}">
            <Method>{operation_name}</Method>
        </MapItem>
        </MapItems>
        }}

        }}
        """
        
        self.generated_classes[class_name] = class_content
        return class_content
    
    def write_message_class(self, message_name, schema_info):
        """
        Generate an IRIS Message class
        
        Args:
            message_name (str): Name of the message class
            schema_info (dict): Information about the schema
            
        Returns:
            str: The generated class content
        """
        class_name = self._sanitize_class_name(f"ms{message_name}")
        
        class_content = f"""Class {class_name} Extends Ens.Request
        {{

        """
        
        # Add properties based on schema
        if isinstance(schema_info, dict) and "properties" in schema_info:
            for prop_name, prop_info in schema_info["properties"].items():
                prop_type = self._map_schema_type_to_iris(prop_info.get("type", "string"))
                class_content += f"Property {prop_name} As {prop_type};\n\n"
        
        class_content += "}\n"
        
        self.generated_classes[class_name] = class_content
        return class_content
    
    def export_classes(self):
        """
        Export all generated classes to .cls files
        
        Returns:
            dict: Status of export operation
        """
        results = {}
        
        if not self.generated_classes:
            return {"status": "warning", "message": "No classes to export"}
        
        for class_name, class_content in self.generated_classes.items():
            file_path = os.path.join(self.output_dir, f"{class_name}.cls")
            
            try:
                # Ensure the directory exists (including package directories)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write the file with proper encoding
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(class_content)
                
                results[class_name] = {
                    "status": "success",
                    "path": file_path,
                    "size": os.path.getsize(file_path)
                }
            except Exception as e:
                results[class_name] = {
                    "status": "error",
                    "error": str(e),
                    "path": file_path
                }
        
        return results
    
    def validate_classes(self):
        """
        Validate the generated IRIS classes for syntax and structural correctness
        
        Returns:
            dict: Validation results
        """
        validation_results = {}
        
        for class_name, class_content in self.generated_classes.items():
            issues = []
            
            # Basic validation checks
            if not "Class " in class_content:
                issues.append("Missing Class declaration")
            
            if not "Extends " in class_content:
                issues.append("Missing Extends keyword")
            
            # Add more validation as needed
            
            validation_results[class_name] = {
                "valid": len(issues) == 0,
                "issues": issues
            }
        
        return validation_results

    def _sanitize_class_name(self, name):
        """Sanitize a name to be valid as an IRIS class name"""
        if not name:
            raise ValueError("Class name cannot be empty")
        if len(name) > 255:  # Example max length
            raise ValueError("Class name too long")

        # Replace non-alphanumeric characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure first character is a letter
        if not name[0].isalpha():
            name = "X" + name
        
        return name
    
    def _map_schema_type_to_iris(self, schema_type):
        """Map OpenAPI schema type to IRIS type"""
        type_mapping = {
            "string": "%String",
            "integer": "%Integer",
            "number": "%Float",
            "boolean": "%Boolean",
            "array": "%Library.ListOfDataTypes",
            "object": "%DynamicObject"
        }
        
        return type_mapping.get(schema_type, "%String")

def sanitize_filename(name):
    """
    Sanitize a string to be used as a filename
    
    Args:
        name (str): The input string
        
    Returns:
        str: A sanitized filename
    """
    # Remove invalid characters for filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Ensure it doesn't start with a space or period
    if name.startswith(' ') or name.startswith('.'):
        name = 'x' + name
    
    return name

class OpenAPIParser:
    """Tool for parsing and analyzing OpenAPI v3 specifications"""

    def analyze(self, openapi_spec):
        """
        Analyzes an OpenAPI specification and returns structured information
        
        Args:
            openapi_spec (Union[dict, str]): The OpenAPI specification as a Python dictionary or JSON string
            
        Returns:
            dict: Structured analysis of the OpenAPI specification
        """
        if isinstance(openapi_spec, str):
            try:
                openapi_spec = json.loads(openapi_spec)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON string provided: {str(e)}"}
        elif not isinstance(openapi_spec, dict):
            return {"error": "Input must be either a JSON string or a dictionary"}

        try:
            result = {
                "info": self._extract_info(openapi_spec),
                "endpoints": self._extract_endpoints(openapi_spec),
                "schemas": self._extract_schemas(openapi_spec)
            }
            return result
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _extract_info(self, spec):
        """Extract basic information from the API spec"""
        info = spec.get("info", {})
        return {
            "title": info.get("title", "Unknown API"),
            "version": info.get("version", "1.0.0"),
            "description": info.get("description", "")
        }
    
    def _extract_endpoints(self, spec):
        """Extract endpoint details from the paths section"""
        paths = spec.get("paths", {})
        endpoints = []
        
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    endpoint = {
                        "path": path,
                        "method": method.upper(),
                        "operationId": operation.get("operationId", f"{method}_{path.replace('/', '_')}"),
                        "summary": operation.get("summary", ""),
                        "description": operation.get("description", ""),
                        "parameters": operation.get("parameters", []),
                        "requestBody": operation.get("requestBody", None),
                        "responses": operation.get("responses", {})
                    }
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _extract_schemas(self, spec):
        """Extract schema definitions"""
        components = spec.get("components", {})
        schemas = components.get("schemas", {})
        
        return {name: details for name, details in schemas.items()}

class GetProductionDetailsTool(BaseTool):
    name: str = "get_production_details"
    description: str = "Get the production details and configuration"
    
    def _run(self, production_name: Optional[str] = None) -> str:
        try:
            # Log the start of getting production details
            logger.info("Getting production details")
            
            # Get production details from ProductionData
            production_info = {
                "name": production_name or "DefaultProduction",
                "timestamp": datetime.now().isoformat(),
                "status": "initializing"
            }
            
            return json.dumps({
                "status": "success",
                "production_info": production_info,
                "message": "Production details retrieved successfully"
            })
            
        except Exception as e:
            logger.error(f"Error getting production details: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Failed to get production details: {str(e)}"
            })

class AnalyzeOpenAPIToolInput(BaseModel):
    openapi_spec: Union[str, Dict[str, Any]] = Field(
        description="OpenAPI specification as JSON string or dictionary"
    )

class ExtractAPISpecTool(BaseTool):
    name: str = "extract_api_spec"
    description: str = "Extract API specifications from input"
    
    def _run(self, user_input: str) -> str:
        # Implementation
        pass

class ValidateAPISpecTool(BaseTool):
    name: str = "validate_api_spec"
    description: str = "Validate API specifications"
    
    def _run(self, api_spec: Dict) -> str:
        # Implementation
        pass

class InteractionTool(BaseTool):
    name: str = "interaction"
    description: str = "Handle user interactions"
    
    def _run(self, missing_info: List[str]) -> str:
        # Implementation
        pass

class TransformToOpenAPITool(BaseTool):
    name: str = "transform_to_openapi"
    description: str = "Transform API specs to OpenAPI format"
    
    def _run(self, validated_spec: Dict) -> str:
        # Implementation
        pass

class ReviewOpenAPITool(BaseTool):
    name: str = "review_openapi"
    description: str = "Review OpenAPI documentation"
    
    def _run(self, openapi_spec: Dict) -> str:
        # Implementation
        pass

class AnalyzeOpenAPITool(BaseTool):
    name: str = "analyze_openapi"
    description: str = "Analyzes an OpenAPI specification and returns structured information"
    input_schema: Type[BaseModel] = AnalyzeOpenAPIToolInput

    def _run(self, openapi_spec: Union[str, Dict[str, Any]]) -> str:
        """
        Analyzes an OpenAPI specification and returns structured information
        
        Args:
            openapi_spec: The OpenAPI specification as a JSON string or dictionary
            
        Returns:
            str: JSON string containing structured analysis
        """
        parser = OpenAPIParser()
        
        try:
            # If input is string, try to parse it as JSON
            if isinstance(openapi_spec, str):
                try:
                    spec_dict = json.loads(openapi_spec)
                except json.JSONDecodeError:
                    return json.dumps({"error": "Invalid JSON string provided"})
            else:
                # If it's already a dictionary, use it directly
                spec_dict = openapi_spec
            
            result = parser.analyze(spec_dict)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Analysis failed: {str(e)}"})

class GenerateProductionClassTool(BaseTool):
    name: str = "generate_production_class"
    description: str = "Generate an IRIS Production class"

    def _run(self, production_name: str, components: str) -> str:
        writer = IRISClassWriter()
        try:
            components_list = json.loads(components)
            return writer.write_production_class(production_name, components_list)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for components"

class CollectGeneratedFilesToolInput(BaseModel):
    directory: str = Field(
        description="Directory containing the generated .cls files"
    )

class CollectGeneratedFilesTool(BaseTool):
    name: str = "collect_generated_files"
    description: str = "Collect all generated IRIS class files into a JSON collection"
    
    def _run(self, directory: str) -> str:
        try:
            collected_files = {}
            
            # Walk through the directory and collect all .cls files
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.cls'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            collected_files[file] = f.read()
            
            return json.dumps({
                "status": "success",
                "message": f"Collected {len(collected_files)} class files",
                "files": collected_files
            })
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error collecting files: {str(e)}",
                "files": {}
            })

class GenerateProductionClassToolInput(BaseModel):
    production_name: str = Field(description="Name of the production")
    components: Union[str, List[Dict[str, Any]]] = Field(description="Components as JSON string or list of dictionaries")

class GenerateBusinessServiceToolInput(BaseModel):
    service_name: str = Field(description="Name of the business service")
    endpoint_info: Union[str, Dict[str, Any]] = Field(description="Endpoint information as JSON string or dictionary")

class GenerateBusinessOperationToolInput(BaseModel):
    operation_name: str = Field(description="Name of the business operation")
    endpoint_info: Union[str, Dict[str, Any]] = Field(description="Endpoint information as JSON string or dictionary")

class GenerateMessageClassToolInput(BaseModel):
    message_name: str = Field(description="Name of the message class")
    schema_info: Union[str, Dict[str, Any]] = Field(description="Schema information as JSON string or dictionary")

class GenerateBusinessServiceTool(BaseTool):
    name: str = "generate_business_service"
    description: str = "Generate an IRIS Business Service class"
    input_schema: Type[BaseModel] = GenerateBusinessServiceToolInput

    def _run(self, service_name: str, endpoint_info: Union[str, Dict[str, Any]]) -> str:
        writer = IRISClassWriter()
        try:
            if isinstance(endpoint_info, str):
                try:
                    endpoint_dict = json.loads(endpoint_info)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format for endpoint info"
            else:
                endpoint_dict = endpoint_info

            class_content = writer.write_business_service(service_name, endpoint_dict)
            # Store the generated class
            writer.generated_classes[f"BS.{service_name}"] = class_content
            return class_content
        except Exception as e:
            return f"Error generating business service: {str(e)}"

class GenerateBusinessOperationTool(BaseTool):
    name: str = "generate_business_operation"
    description: str = "Generate an IRIS Business Operation class"
    input_schema: Type[BaseModel] = GenerateBusinessOperationToolInput

    def _run(self, operation_name: str, endpoint_info: Union[str, Dict[str, Any]]) -> str:
        writer = IRISClassWriter()
        try:
            if isinstance(endpoint_info, str):
                try:
                    endpoint_dict = json.loads(endpoint_info)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format for endpoint info"
            else:
                endpoint_dict = endpoint_info

            class_content = writer.write_business_operation(operation_name, endpoint_dict)
            # Store the generated class
            writer.generated_classes[f"BO.{operation_name}"] = class_content
            return class_content
        except Exception as e:
            return f"Error generating business operation: {str(e)}"

class GenerateMessageClassTool(BaseTool):
    name: str = "generate_message_class"
    description: str = "Generate an IRIS Message class"
    input_schema: Type[BaseModel] = GenerateMessageClassToolInput

    def _run(self, message_name: str, schema_info: Union[str, Dict[str, Any]]) -> str:
        writer = IRISClassWriter()
        try:
            if isinstance(schema_info, str):
                try:
                    schema_dict = json.loads(schema_info)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format for schema info"
            else:
                schema_dict = schema_info

            class_content = writer.write_message_class(message_name, schema_dict)
            # Store the generated class
            writer.generated_classes[f"MSG.{message_name}"] = class_content
            return class_content
        except Exception as e:
            return f"Error generating message class: {str(e)}"

class GenerateProductionClassTool(BaseTool):
    name: str = "generate_production_class"
    description: str = "Generate an IRIS Production class"
    input_schema: Type[BaseModel] = GenerateProductionClassToolInput

    def _run(self, production_name: str, components: Union[str, List[Dict[str, Any]]]) -> str:
        writer = IRISClassWriter()
        try:
            if isinstance(components, str):
                try:
                    components_list = json.loads(components)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format for components"
            else:
                components_list = components

            class_content = writer.write_production_class(production_name, components_list)
            # Store the generated class
            writer.generated_classes[f"Production.{production_name}"] = class_content
            return class_content
        except Exception as e:
            return f"Error generating production class: {str(e)}"

class ExportIRISClassesToolInput(BaseModel):
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional output directory path. If not provided, will use default directory"
    )

class ExportIRISClassesTool(BaseTool):
    name: str = "export_iris_classes"
    description: str = "Export all generated classes to .cls files"
    input_schema: Type[BaseModel] = ExportIRISClassesToolInput

    def _run(self, output_dir: Optional[str] = None) -> str:
        writer = IRISClassWriter()
        try:
            if not writer.generated_classes:
                return json.dumps({
                    "status": "warning",
                    "message": "No classes to export",
                    "details": "The generated_classes dictionary is empty. Make sure classes were generated successfully before exporting."
                })

            if output_dir:
                writer.output_dir = os.path.abspath(output_dir)
                writer._ensure_output_directory()

            results = writer.export_classes()
            return json.dumps({
                "status": "success",
                "message": f"Exported {len(writer.generated_classes)} classes",
                "details": results
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

class ValidateIRISClassesToolInput(BaseModel):
    class_names: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific class names to validate. If not provided, validates all classes"
    )

class ValidateIRISClassesTool(BaseTool):
    name: str = "validate_iris_classes"
    description: str = "Validate the generated IRIS classes"
    input_schema: Type[BaseModel] = ValidateIRISClassesToolInput

    def _run(self, class_names: Optional[List[str]] = None) -> str:
        """
        Validate the generated IRIS classes
        
        Args:
            class_names (Optional[List[str]]): Optional list of specific class names to validate
            
        Returns:
            str: JSON string containing validation results
        """
        writer = IRISClassWriter()
        try:
            results = writer.validate_classes(class_names)
            return json.dumps(results, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

# Create tool instances
analyze_openapi_tool = AnalyzeOpenAPITool()
generate_production_class_tool = GenerateProductionClassTool()
generate_business_service_tool = GenerateBusinessServiceTool()
generate_business_operation_tool = GenerateBusinessOperationTool()
generate_message_class_tool = GenerateMessageClassTool()
export_iris_classes_tool = ExportIRISClassesTool()
validate_iris_classes_tool = ValidateIRISClassesTool()
collect_generated_files_tool = CollectGeneratedFilesTool()


class APIAgents:
    def __init__(self, llm):
        self.logger = logging.getLogger('facilis.APIAgents')
        self.llm = llm
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

    def create_analyzer_agent(self) -> Agent:
        return Agent(
            role="OpenAPI Specification Analyzer",
            goal="Thoroughly analyze OpenAPI specifications and plan IRIS Interoperability components",
            backstory="""You are an expert in both OpenAPI specifications and InterSystems IRIS Interoperability. 
            Your job is to analyze OpenAPI documents and create a detailed plan for how they should be 
            implemented as IRIS Interoperability components.""",
            verbose=True,
            allow_delegation=False,
            tools=[analyze_openapi_tool],
            llm=self.llm
        )

    def create_bs_generator_agent(self) -> Agent:
        return Agent(
            role="IRIS Production and Business Service Generator",
            goal="Generate properly formatted IRIS Production and Business Service classes from OpenAPI specifications",
            backstory="""You are an experienced InterSystems IRIS developer specializing in Interoperability Productions.
            Your expertise is in creating Business Services and Productions that can receive and process incoming requests based on
            API specifications.""",
            verbose=True,
            allow_delegation=True,
            tools=[generate_production_class_tool, generate_business_service_tool],
            llm=self.llm
        )

    def create_bo_generator_agent(self) -> Agent:
        return Agent(
            role="IRIS Business Operation Generator",
            goal="Generate properly formatted IRIS Business Operation classes from OpenAPI specifications",
            backstory="""You are an experienced InterSystems IRIS developer specializing in Interoperability Productions.
            Your expertise is in creating Business Operations that can send requests to external systems
            based on API specifications.""",
            verbose=True,
            allow_delegation=True,
            tools=[generate_business_operation_tool, generate_message_class_tool],
            llm=self.llm
        )

    def create_exporter_agent(self) -> Agent:
        return Agent(
            role="IRIS Class Exporter",
            goal="Export and validate IRIS class definitions to proper .cls files",
            backstory="""You are an InterSystems IRIS deployment specialist. Your job is to ensure 
            that generated IRIS class definitions are properly exported as valid .cls files that 
            can be directly imported into an IRIS environment.""",
            verbose=True,
            allow_delegation=False,
            tools=[export_iris_classes_tool, validate_iris_classes_tool],
            llm=self.llm
        )
        
    def create_collector_agent(self) -> Agent:
        return Agent(
            role="IRIS Class Collector",
            goal="Collect all generated IRIS class files into a JSON collection",
            backstory="""You are a file system specialist responsible for gathering and 
            organizing generated IRIS class files into a structured collection.""",
            verbose=True,
            allow_delegation=False,
            tools=[CollectGeneratedFilesTool()],
            llm=self.llm
        )

class APISpecificationCrew:
    def __init__(self, llm, production_data: ProductionData, callback=None):
        self.callback = callback
        self.current_agent = None 
        api_agents = APIAgents(llm)
        self.production_agent = api_agents.create_production_agent()
        self.interaction_agent = api_agents.create_interaction_agent()
        self.validation_agent = api_agents.create_validation_agent()
        self.extraction_agent = api_agents.create_extraction_agent()
        self.transformation_agent = api_agents.create_transformation_agent()
        self.reviewer_agent = api_agents.create_reviewer_agent()
        self.iris_i14y_agent = api_agents.create_iris_i14y_agent()
        self.production_data = production_data
        self.analyzer_agent = api_agents.create_analyzer_agent()
        self.bs_generator_agent = api_agents.create_bs_generator_agent()
        self.bo_generator_agent = api_agents.create_bo_generator_agent()
        self.exporter_agent = api_agents.create_exporter_agent()
        self.collector_agent = api_agents.create_collector_agent()

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

    def analysis_task(self, openapi_spec: Dict) -> Task:
        return Task(
            description="""Analyze the OpenAPI specification and plan the necessary IRIS Interoperability components. 
            Include a list of all components that should be in the Production class.""",
            agent=self.analyzer_agent,
            expected_output="A detailed analysis of OpenAPI spec and plan for IRIS components",
            input={
                "openapi_spec": openapi_spec,
                "production_name": self.production_data.current_production_name
            }
        )

    def bs_generation_task(self) -> Task:
        return Task(
            description="Generate Business Service classes based on the OpenAPI endpoints",
            agent=self.bs_generator,
            expected_output="IRIS Business Service class definitions",
            context=[self.analysis_task]
        )

    def bo_generation_task(self) -> Task:
        return Task(
            description="Generate Business Operation classes based on the OpenAPI endpoints",
            agent=self.bo_generator,
            expected_output="IRIS Business Operation class definitions",
            context=[self.analysis_task]
        )

    def export_task(self) -> Task:
        return Task(
            description="Export all generated IRIS classes as valid .cls files",
            agent=self.exporter,
            expected_output="Valid IRIS .cls files saved to output directory",
            context=[self.bs_generation_task, self.bo_generation_task],
            input={
                "output_dir": OUTPUT_DIR
            }
        )

    def validate_task(self) -> Task:
        return Task(
            description="Validate all generated IRIS classes",
            agent=self.exporter,
            expected_output="Validation results for all generated classes",
            context=[self.export_task],
            input={
                "class_names": None  # Optional, will validate all classes if not specified
            }
        )

    def production_generation_task(self) -> Task:
        return Task(
            description="Generate the Production class that includes all generated components",
            agent=self.bs_generator,  # We can use the bs_generator since it has the generate_production_class_tool
            expected_output="IRIS Production class definition",
            context=[self.bs_generation_task, self.bo_generation_task],  # This ensures it runs after BS and BO generation
        )

    def production_generation_task(self) -> Task:
        return Task(
            description="Generate the Production class that includes all generated components",
            agent=self.bs_generator,  # We can use the bs_generator since it has the generate_production_class_tool
            input={
                "production_name": "${production_name}"  # Add production name input
            },
            expected_output="IRIS Production class definition",
            context=[self.bs_generation_task, self.bo_generation_task],  # This ensures it runs after BS and BO generation
        )

    def collection_task(self) -> Task:
        return Task(
            description="Collect all generated IRIS class files into a JSON collection",
            agent=self.collector,
            expected_output="JSON collection of all generated .cls files",
            context=[self.export_task, self.validate_task],
            input={
                "directory": OUTPUT_DIR
            }
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
    from crewai import LLM
    logger = logging.getLogger('facilis.get_facilis_llm')
    
    ai_engine = os.getenv("AI_ENGINE")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME")
    
    logger.info(f"Initializing LLM with engine: {ai_engine}")

    if ai_engine == "openai":
        from openai import OpenAI
        logger.info(f"Using OpenAI with model: {model_name}")
        os.environ["OPENAI_API_KEY"] = api_key
        return LLM(model=model_name, temperature=0)  # Return just the model name

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
        return LLM(model=model_name, temperature=0)

    if ai_engine == "gemini":
        os.environ["GOOGLE_API_KEY"] = api_key
        return LLM(model=model_name, temperature=0)
    
    if ai_engine == "ollama":
        return LLM(model=model_name, temperature=0)
    
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

class CrewFactory:
    @staticmethod
    def create_crew(production_data: ProductionData, callback: Optional[StreamlitCallback] = None) -> 'CrewManager':
        return CrewManager(production_data, callback)

class CrewManager:
    def __init__(self, production_data: ProductionData, callback: Optional[StreamlitCallback] = None):
        self.production_data = production_data
        self.callback = callback
        self._init_agents()

    def _init_agents(self):
        """Initialize all agents with their roles and tools"""
        # Production Agent
        self.production_agent = Agent(
            role="production",
            goal="Get production details and setup",
            backstory="I am responsible for getting production details and initial setup",
            tools=[GetProductionDetailsTool()]
        )

        # Extraction Agent
        self.extraction_agent = Agent(
            role="extraction",
            goal="Extract API specifications from input",
            backstory="I extract and structure API specifications from user input",
            tools=[ExtractAPISpecTool()]
        )

        # Validation Agent
        self.validation_agent = Agent(
            role="validation",
            goal="Validate API specifications",
            backstory="I validate the extracted API specifications for completeness and correctness",
            tools=[ValidateAPISpecTool()]
        )

        # Interaction Agent
        self.interaction_agent = Agent(
            role="interaction",
            goal="Handle missing information and user interaction",
            backstory="I handle user interactions and gather missing information",
            tools=[InteractionTool()]
        )

        # Transformation Agent
        self.transformation_agent = Agent(
            role="transformation",
            goal="Transform API specs to OpenAPI format",
            backstory="I transform the validated specifications into OpenAPI format",
            tools=[TransformToOpenAPITool()]
        )

        # Reviewer Agent
        self.reviewer_agent = Agent(
            role="reviewer",
            goal="Review OpenAPI documentation",
            backstory="I review and validate the OpenAPI documentation",
            tools=[ReviewOpenAPITool()]
        )

        # Analyzer Agent
        self.analyzer_agent = Agent(
            role="analyzer",
            goal="Analyze OpenAPI structure",
            backstory="I analyze the OpenAPI structure for IRIS component generation",
            tools=[AnalyzeOpenAPITool()]
        )

        # Business Service Generator Agent
        self.bs_generator_agent = Agent(
            role="bs_generator",
            goal="Generate Business Service components",
            backstory="I generate IRIS Business Service components",
            tools=[GenerateBusinessServiceTool()]
        )

        # Business Operation Generator Agent
        self.bo_generator_agent = Agent(
            role="bo_generator",
            goal="Generate Business Operation components",
            backstory="I generate IRIS Business Operation components",
            tools=[GenerateBusinessOperationTool()]
        )

        # Exporter Agent
        self.exporter_agent = Agent(
            role="exporter",
            goal="Export generated classes",
            backstory="I export the generated IRIS classes",
            tools=[ExportIRISClassesTool()]
        )

        # Collector Agent
        self.collector_agent = Agent(
            role="collector",
            goal="Collect generated files",
            backstory="I collect all generated files and prepare them for delivery",
            tools=[CollectGeneratedFilesTool()]
        )

    def get_production_details(self) -> Task:
        return Task(
            description="Getting production details",
            agent=self.production_agent,
            expected_output="Production details and configuration",
            context=[]  # Empty list for context
        )

    def extraction_task(self, user_input: str) -> Task:
        return Task(
            description=f"Extract API specifications from: {user_input}",
            agent=self.extraction_agent,
            expected_output="Extracted API specifications",
            context=[{
                "input": user_input,
                "type": "user_input"
            }]
        )

    def validation_task(self, api_spec: Dict) -> Task:
        return Task(
            description="Validate the extracted API specifications",
            agent=self.validation_agent,
            expected_output="Validation results",
            context=[{
                "spec": api_spec,
                "type": "api_spec"
            }]
        )

    def interaction_task(self, missing_info: List[str]) -> Task:
        return Task(
            description="Gather missing information through user interaction",
            agent=self.interaction_agent,
            expected_output="Completed information",
            context=[{
                "missing": missing_info,
                "type": "missing_info"
            }]
        )

    def transformation_task(self, validated_spec: Dict) -> Task:
        return Task(
            description="Transform specifications to OpenAPI format",
            agent=self.transformation_agent,
            expected_output="OpenAPI specification",
            context=[{
                "spec": validated_spec,
                "type": "validated_spec"
            }]
        )

    def review_task(self, openapi_spec: Dict) -> Task:
        return Task(
            description="Review OpenAPI documentation",
            agent=self.reviewer_agent,
            expected_output="Review results",
            context=[{
                "spec": openapi_spec,
                "type": "openapi_spec"
            }]
        )

    def analysis_task(self, openapi_spec: Dict) -> Task:
        return Task(
            description="Analyze OpenAPI structure for IRIS components",
            agent=self.analyzer_agent,
            expected_output="Analysis results",
            context=[{
                "spec": openapi_spec,
                "type": "openapi_spec"
            }]
        )

    def bs_generation_task(self) -> Task:
        return Task(
            description="Generate Business Service components",
            agent=self.bs_generator_agent,
            expected_output="Generated Business Service classes",
            context=[]
        )

    def bo_generation_task(self) -> Task:
        return Task(
            description="Generate Business Operation components",
            agent=self.bo_generator_agent,
            expected_output="Generated Business Operation classes",
            context=[]
        )

    def export_task(self) -> Task:
        return Task(
            description="Export generated IRIS classes",
            agent=self.exporter_agent,
            expected_output="Exported class files",
            context=[]
        )

    def collection_task(self) -> Task:
        return Task(
            description="Collect all generated files",
            agent=self.collector_agent,
            expected_output="Collection of generated files",
            context=[]
        )




async def process_api_integration(
    user_input: str, 
    production_data: ProductionData, 
    st_container=None
) -> Dict:
    """Process the API integration workflow"""
    logger = logging.getLogger('facilis.process_api_integration')
    logger.info("Starting API integration process")
    
    try:
        callback = StreamlitCallback(st_container)
        crew = CrewFactory.create_crew(
            production_data=production_data,
            callback=callback
        )
        
        # First task: Get production details
        initial_crew = Crew(
            agents=[crew.production_agent],
            tasks=[crew.get_production_details()],
            process_callbacks=[callback],
            verbose=True
        )
        
        logger.info("Getting production details")
        production_result = initial_crew.kickoff()
        
        # Extract API specs
        extraction_crew = Crew(
            agents=[crew.extraction_agent],
            tasks=[crew.extraction_task(user_input)],
            process_callbacks=[callback],
            verbose=True
        )
        
        extraction_result = extraction_crew.kickoff()
        extracted_specs = json.loads(extract_json_from_markdown(extraction_result))
        
        # Continue with validation and other tasks
        if extracted_specs:
            validation_crew = Crew(
                agents=[
                    crew.validation_agent,
                    crew.interaction_agent,
                    crew.transformation_agent,
                    crew.reviewer_agent,
                    crew.analyzer_agent
                ],
                tasks=[crew.validation_task(extracted_specs)],
                process_callbacks=[callback],
                verbose=True
            )
            
            result = validation_crew.kickoff()
            review_result = json.loads(extract_json_from_markdown(result))
            
            if review_result.get("is_valid", False):
                # Generation tasks
                generation_tasks = [
                    crew.bs_generation_task(),
                    crew.bo_generation_task(),
                    crew.export_task(),
                    crew.collection_task()
                ]
                
                final_result = None
                for task in generation_tasks:
                    task_crew = Crew(
                        agents=[task.agent],
                        tasks=[task],
                        process_callbacks=[callback],
                        verbose=True
                    )
                    final_result = task_crew.kickoff()
                
                collection_result = json.loads(extract_json_from_markdown(final_result))
                
                return {
                    "review_details": review_result,
                    "openapi_documentation": review_result.get("approved_spec"),
                    "generated_files": collection_result.get("files", {})
                }
            
            return {
                "review_details": review_result,
                "error": "API specification validation failed"
            }
            
    except Exception as e:
        logger.error(f"Error in process_api_integration: {str(e)}")
        raise

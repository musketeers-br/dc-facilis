import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# Initialize OpenAI GPT-4o-mini model
llm = ChatOpenAI(model_name=LLM_MODEL_NAME , temperature=0, openai_api_key=OPENAI_API_KEY)

def interaction_agent(missing_fields):
    """
    Interacts with the user via Streamlit to obtain missing fields.
    """
    return f"Missing fields: {', '.join(missing_fields)}. Please provide them."

def production_agent():
    """Asks user whether to use an existing production or create a new one, including namespace."""
    prompt = """
    The user must choose whether to use an existing production or create a new one.
    Ask them to provide:
    - Production Name
    - Namespace

    If they are creating a new production, confirm before proceeding.
    """
    response = llm.predict(prompt)
    return response

def validation_agent(extracted_data):
    """
    Validates extracted API specifications to ensure correctness and consistency.
    """
    errors = []
    
    # Validate host
    if not re.match(r"^([a-zA-Z0-9.-]+)$", extracted_data.get("host", "")):
        errors.append("Invalid host format.")
    
    # Validate endpoint
    if not extracted_data.get("endpoint", "").startswith("/"):
        errors.append("Endpoint must start with '/'.")
    
    # Validate HTTP method
    valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
    if extracted_data.get("HTTP_Method", "").upper() not in valid_methods:
        errors.append("Invalid HTTP method.")
    
    # Validate port (if provided)
    port = extracted_data.get("port")
    if port and (not str(port).isdigit() or int(port) < 1 or int(port) > 65535):
        errors.append("Invalid port number.")
    
    # Validate JSON model for certain methods
    json_model = extracted_data.get("json_model")
    if extracted_data.get("HTTP_Method", "").upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        if not json_model:
            errors.append("Missing JSON model for method that requires a request body.")
        else:
            try:
                json.loads(json_model)
            except json.JSONDecodeError:
                errors.append("Invalid JSON model.")
    
    if errors:
        return {"error": "Validation failed", "issues": errors}
    return {"status": "Valid API specification"}

def transformation_agent(validated_endpoints, production_name, namespace):
    """
    Converts structured API details into a single OpenAPI-compliant JSON.
    """
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": production_name,
            "version": "1.0.0",
            "namespace": namespace
        },
        "paths": {}
    }
    
    for endpoint_data in validated_endpoints:
        endpoint = endpoint_data["endpoint"]
        method = endpoint_data["HTTP_Method"].lower()
        
        if endpoint not in openapi_spec["paths"]:
            openapi_spec["paths"][endpoint] = {}
        
        openapi_spec["paths"][endpoint][method] = {
            "summary": "Auto-generated endpoint",
            "parameters": [
                {"name": param, "in": "query", "required": False, "schema": {"type": "string"}}
                for param in endpoint_data.get("params", [])
            ],
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "example": endpoint_data.get("json_model", {})
                        }
                    }
                }
            }
        }
    
    return json.dumps(openapi_spec, indent=2)

def extract_api_specs(descriptions):
    """
    Extracts multiple API specifications from natural language descriptions.
    """
    extracted_endpoints = []
    missing_fields = []
    
    for description in descriptions:
        prompt = (
            "You are an assistant specialized in identifying API specifications from natural language descriptions. "
            "Given the following user input, extract the necessary information and return a structured JSON with the fields: "
            "host (required), endpoint (required), HTTP_Method (required), params (optional parameters), "
            "port (if available), json_model (if applicable for POST, PUT, PATCH, DELETE), and authentication (if applicable). "
            "If any essential field (host, endpoint, or HTTP method) is missing, return 'missing'. "
            "If the HTTP method is POST, PUT, PATCH, or DELETE, ensure a JSON model is provided. "
            "If missing, mark it as 'missing' so the interaction agent can request it from the user.\n\n"
            f"Input: {description}\n"
            "Expected output in JSON:"
        )
        
        response = llm([SystemMessage(content=prompt)])
        
        try:
            extracted_data = json.loads(response.content)
            if any(value == 'missing' for value in extracted_data.values()):
                missing_fields.extend([key for key, value in extracted_data.items() if value == 'missing'])
            extracted_endpoints.append(extracted_data)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response"}
    
    if missing_fields:
        return interaction_agent(missing_fields)
    
    validated_endpoints = []
    for endpoint_data in extracted_endpoints:
        validation_result = validation_agent(endpoint_data)
        if "error" in validation_result:
            return validation_result
        validated_endpoints.append(endpoint_data)
    
    return validated_endpoints

Class dc.facilis.core Extends %RegisteredObject
{

    /// Crew excution start
    ClassMethod Execute() As %Status
    {
        Set sc = $$$OK
        Return sc
    }

    /// Agent
ClassMethod Agents(prompt as %String) As %String [ Language = python]
{
    import os
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    from dotenv import load_dotenv
    import re
    import json

    # Carregar variáveis de ambiente
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Inicializar o modelo OpenAI GPT-4o-mini
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

    def interaction_agent(missing_fields):
        """
        Interacts with the user to obtain the missing fields.
        """
        print(f"The following fields are missing: {', '.join(missing_fields)}")
        user_input = input("Please provide the missing information or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting the application due to missing essential information.")
            exit()
        return user_input

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
        
        # Validate JSON model (if provided)
        json_model = extracted_data.get("json_model")
        if json_model:
            try:
                json.loads(json_model)
            except json.JSONDecodeError:
                errors.append("Invalid JSON model.")
        
        if errors:
            return {"error": "Validation failed", "issues": errors}
        return {"status": "Valid API specification"}

    def transformation_agent(validated_data):
        """
        Converts structured API details into OpenAPI-compliant JSON.
        """
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Generated API",
                "version": "1.0.0"
            },
            "paths": {
                validated_data["endpoint"]: {
                    validated_data["HTTP_Method"].lower(): {
                        "summary": "Auto-generated endpoint",
                        "parameters": [
                            {"name": param, "in": "query", "required": False, "schema": {"type": "string"}}
                            for param in validated_data.get("params", [])
                        ],
                        "responses": {
                            "200": {
                                "description": "Successful response",
                                "content": {
                                    "application/json": {
                                        "example": validated_data.get("json_model", {})
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return json.dumps(openapi_spec, indent=2)

    def reviewer_agent(openapi_json):
        """
        Reviews OpenAPI JSON to ensure compliance with OpenAPI 3.0 specifications.
        """
        try:
            openapi_data = json.loads(openapi_json)
            # Check required fields in OpenAPI structure
            required_keys = ["openapi", "info", "paths"]
            for key in required_keys:
                if key not in openapi_data:
                    return {"error": f"Missing required OpenAPI key: {key}"}
            
            # Validate paths
            if not isinstance(openapi_data["paths"], dict) or not openapi_data["paths"]:
                return {"error": "Invalid or missing paths in OpenAPI specification"}
            
            return {"status": "OpenAPI specification is valid and ready for deployment"}
        except json.JSONDecodeError:
            return {"error": "Invalid OpenAPI JSON format"}

    def extract_api_spec(description: str):
        """
        Extracts API specifications from a natural language description.
        """
        prompt = (
            "You are an assistant specialized in identifying API specifications from natural language descriptions. "
            "Given the following user input, extract the necessary information and return a structured JSON with the fields: "
            "host, endpoint, HTTP_Method, params (optional parameters), and authentication (if applicable). "
            "If any essential field is missing, return 'missing'.\n\n"
            f"Input: {description}\n"
            "Expected output in JSON:"
        )
        
        response = llm([SystemMessage(content=prompt)])
        # rest of the code follows

        try:
            extracted_data = json.loads(response.content)
            # Verificar se há campos ausentes
            missing_fields = [key for key, value in extracted_data.items() if value == 'missing']
            if missing_fields:
                additional_info = interaction_agent(missing_fields)
                extracted_data.update({field: additional_info for field in missing_fields})
            
            # Validate the extracted data
            validation_result = validation_agent(extracted_data)
            if "error" in validation_result:
                return validation_result
            
            openapi_json = transformation_agent(extracted_data)

            # Review OpenAPI compliance
            review_result = reviewer_agent(openapi_json)
            if "error" in review_result:
                return review_result

            return openapi_json

        except json.JSONDecodeError:
            return {"error": "Failed to parse response"}

    # Exemplo de uso
    description = "I want to get all pull requests from octocat/Hello-World on github. The host is api.github.com and endpoint '/repos/{owner}/{repo}/pulls' Use get method and include optional parameters like 'state', 'sort', and 'direction'"
    result = extract_api_spec(description)
    print(result)


    description = "I want to get all information of sqlzilla repository "
    #result = extract_api_spec(description)
    #print(result)    
}
}
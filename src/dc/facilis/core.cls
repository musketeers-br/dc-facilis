Class dc.facilis.core Extends %RegisteredObject
{

/// Execute
ClassMethod Execute(json As %DynamicObject, Output response As %DynamicObject) As %Status
{
    Set tSC = $$$OK
	Try {
        Set response = {}
        Set openapi = json.%Get("openapi_spec")
        Set production = json.%Get("production_name")
        Set:(production="") production = "facilisProduction"
        Set resp = ..GenerateI14y(production, openapi.%ToJSON())
        Set:(resp'="") response = {}.%FromJSON(resp)
	} Catch tException {
		Set:$$$ISOK(tSC) tSC = tException.AsStatus()
	}
	Return tSC
}

/// Generate Interoperability
ClassMethod GenerateI14y(productionName As %String, openApiSpec As %String) As %String [ Language = python ]
{
    import os
    import json
    import re
    from dotenv import load_dotenv
    from crewai import Crew, Agent, Task
    from crewai.tools import BaseTool
    from typing import Union, Dict, Any, Type, List, Optional
    from pydantic import BaseModel, Field

    load_dotenv()

    def get_facilis_llm():
        """Returns the appropriate chat model based on AI_ENGINE selection."""
        
        ai_engine = os.getenv("AI_ENGINE")
        api_key = os.getenv("API_KEY")
        model_name = os.getenv("LLM_MODEL_NAME")
        
        if ai_engine == "openai":
            from openai import OpenAI
            os.environ["OPENAI_API_KEY"] = api_key
            return model_name  # Return just the model name

        if ai_engine in ["azureopenai", "azure_openai"]:
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
            if not azure_endpoint or not azure_deployment_name:
                raise ValueError("Azure OpenAI requires AZURE_ENDPOINT and AZURE_DEPLOYMENT_NAME in .env")
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
        return None

    class AnalyzeOpenAPIToolInput(BaseModel):
        openapi_spec: Union[str, Dict[str, Any]] = Field(
            description="OpenAPI specification as JSON string or dictionary"
        )

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

    def create_analyzer_agent():
        return Agent(
            role="OpenAPI Specification Analyzer",
            goal="Thoroughly analyze OpenAPI specifications and plan IRIS Interoperability components",
            backstory="""You are an expert in both OpenAPI specifications and InterSystems IRIS Interoperability. 
            Your job is to analyze OpenAPI documents and create a detailed plan for how they should be 
            implemented as IRIS Interoperability components.""",
            verbose=True,
            allow_delegation=False,
            tools=[analyze_openapi_tool],
            llm=get_facilis_llm()
        )

    def create_bs_generator_agent():
        return Agent(
            role="IRIS Production and Business Service Generator",
            goal="Generate properly formatted IRIS Production and Business Service classes from OpenAPI specifications",
            backstory="""You are an experienced InterSystems IRIS developer specializing in Interoperability Productions.
            Your expertise is in creating Business Services that can receive and process incoming requests based on
            API specifications.""",
            verbose=True,
            allow_delegation=True,
            tools=[generate_production_class_tool, generate_business_service_tool],
            llm=get_facilis_llm()
        )

    def create_bo_generator_agent():
        return Agent(
            role="IRIS Business Operation Generator",
            goal="Generate properly formatted IRIS Business Operation classes from OpenAPI specifications",
            backstory="""You are an experienced InterSystems IRIS developer specializing in Interoperability Productions.
            Your expertise is in creating Business Operations that can send requests to external systems
            based on API specifications.""",
            verbose=True,
            allow_delegation=True,
            tools=[generate_business_operation_tool, generate_message_class_tool],
            llm=get_facilis_llm()
        )

    def create_exporter_agent():
        return Agent(
            role="IRIS Class Exporter",
            goal="Export and validate IRIS class definitions to proper .cls files",
            backstory="""You are an InterSystems IRIS deployment specialist. Your job is to ensure 
            that generated IRIS class definitions are properly exported as valid .cls files that 
            can be directly imported into an IRIS environment.""",
            verbose=True,
            allow_delegation=False,
            tools=[export_iris_classes_tool, validate_iris_classes_tool],
            llm=get_facilis_llm()
        )

    # Define the tool functions
    def analyze_openapi(openapi_spec: str) -> str:
        """
        Analyzes an OpenAPI specification and returns structured information about paths, 
        operations, and schemas
        
        Args:
            openapi_spec (str): The OpenAPI specification as a JSON string
            
        Returns:
            str: JSON string containing structured analysis of the OpenAPI specification
        """
        parser = OpenAPIParser()
        if isinstance(openapi_spec, str):
            try:
                spec_dict = json.loads(openapi_spec)
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid JSON string provided: {str(e)}"})
        else:
            spec_dict = openapi_spec
        
        result = parser.analyze(spec_dict)
        return json.dumps(result, indent=2)

    def generate_production_class(production_name: str, components: str) -> str:
        """
        Generate an IRIS Production class
        
        Args:
            production_name (str): Name of the production
            components (str): JSON string containing list of components
            
        Returns:
            str: The generated class content
        """
        writer = IRISClassWriter()
        try:
            components_list = json.loads(components)
            return writer.write_production_class(production_name, components_list)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for components"

    def generate_business_service(service_name: str, endpoint_info: str) -> str:
        """
        Generate an IRIS Business Service class
        
        Args:
            service_name (str): Name of the business service
            endpoint_info (str): JSON string containing endpoint information
            
        Returns:
            str: The generated class content
        """
        writer = IRISClassWriter()
        try:
            endpoint_dict = json.loads(endpoint_info)
            return writer.write_business_service(service_name, endpoint_dict)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for endpoint info"

    def generate_business_operation(operation_name: str, endpoint_info: str) -> str:
        """
        Generate an IRIS Business Operation class
        
        Args:
            operation_name (str): Name of the business operation
            endpoint_info (str): JSON string containing endpoint information
            
        Returns:
            str: The generated class content
        """
        writer = IRISClassWriter()
        try:
            endpoint_dict = json.loads(endpoint_info)
            return writer.write_business_operation(operation_name, endpoint_dict)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for endpoint info"

    def generate_message_class(message_name: str, schema_info: str) -> str:
        """
        Generate an IRIS Message class
        
        Args:
            message_name (str): Name of the message class
            schema_info (str): JSON string containing schema information
            
        Returns:
            str: The generated class content
        """
        writer = IRISClassWriter()
        try:
            schema_dict = json.loads(schema_info)
            return writer.write_message_class(message_name, schema_dict)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for schema info"

    def export_iris_classes() -> str:
        """
        Export all generated classes to .cls files
        
        Returns:
            str: JSON string containing status of export operation
        """
        writer = IRISClassWriter()
        results = writer.export_classes()
        return json.dumps(results, indent=2)

    def validate_iris_classes() -> str:
        """
        Validate the generated IRIS classes
        
        Returns:
            str: JSON string containing validation results
        """
        writer = IRISClassWriter()
        results = writer.validate_classes()
        return json.dumps(results, indent=2)


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
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

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    class IRISClassWriter:
        """Tool for generating InterSystems IRIS interoperability class files"""
        _instance = None
        _initialized = False
        
        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(self, output_dir="./output"):
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
        
        def write_business_service(self, service_name, endpoint_info):
            """
            Generate an IRIS Business Service class
            
            Args:
                service_name (str): Name of the business service
                endpoint_info (dict): Information about the API endpoint
                
            Returns:
                str: The generated class content
            """
            class_name = self._sanitize_class_name(f"bs_{service_name}")
            
            class_content = f"""Class {class_name} Extends Ens.BusinessService
            {{

            Parameter ADAPTER = "EnsLib.HTTP.InboundAdapter";

            Method OnProcessInput(pInput As %Stream.Object, Output pOutput As %RegisteredObject) As %Status
            {{
                Set tSC = $$$OK
                Try {{
                    // Process input and create a request message
                    Set tRequest = ##class(ms_{service_name}).%New()
                    
                    // Forward to appropriate business process or operation
                    Set tSC = ..SendRequestSync("to_{service_name}", tRequest, .tResponse)
                    
                    // Handle response
                    If $IsObject(tResponse) {{
                        // Process response
                    }}
                }}
                Catch ex {{
                    Set tSC = ex.AsStatus()
                }}
                
                Return tSC
            }}

            }}
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
            class_name = self._sanitize_class_name(f"bo_{operation_name}")
            
            method = endpoint_info.get("method", "GET")
            path = endpoint_info.get("path", "/")
            
            class_content = f"""Class {class_name} Extends Ens.BusinessOperation
            {{

            Parameter ADAPTER = "EnsLib.HTTP.OutboundAdapter";

            Property Adapter As EnsLib.HTTP.OutboundAdapter;

            Parameter INVOCATION = "Queue";

            Method {operation_name}(pRequest As ms_{operation_name}, Output pResponse As Ens.Response) As %Status
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
            <MapItem MessageType="ms_{operation_name}">
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
            class_name = self._sanitize_class_name(f"ms_{message_name}")
            
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


    # -=--=--=--=--=--=--=--=--=--=--=--=--=--=--=-
    # Initialize tools
    openapi_parser = OpenAPIParser()
    iris_class_writer = IRISClassWriter(output_dir="./output/iris_classes")

    # Create the agents
    analyzer = create_analyzer_agent()
    bs_generator = create_bs_generator_agent()
    bo_generator = create_bo_generator_agent()
    exporter = create_exporter_agent()

    # Create tasks
    analysis_task = Task(
        description="Analyze the OpenAPI specification and plan the necessary IRIS Interoperability components",
        agent=analyzer,
        expected_output="A detailed analysis of OpenAPI spec and plan for IRIS components",
        input={
            "openapi_spec": openApiSpec  # Pass the dictionary directly, the tool will handle it
        }
    )

    bs_generation_task = Task(
        description="Generate Business Service classes based on the OpenAPI endpoints",
        agent=bs_generator,
        expected_output="IRIS Business Service class definitions",
        context=[analysis_task]
    )

    bo_generation_task = Task(
        description="Generate Business Operation classes based on the OpenAPI endpoints",
        agent=bo_generator,
        expected_output="IRIS Business Operation class definitions",
        context=[analysis_task]
    )

    export_task = Task(
        description="Export all generated IRIS classes as valid .cls files",
        agent=exporter,
        expected_output="Valid IRIS .cls files saved to output directory",
        context=[bs_generation_task, bo_generation_task],
        input={
            "output_dir": "./output/iris_classes"  # Optional
        }
    )

    validate_task = Task(
        description="Validate all generated IRIS classes",
        agent=exporter,
        expected_output="Validation results for all generated classes",
        context=[export_task],
        input={
            "class_names": None  # Optional, will validate all classes if not specified
        }
    )

    # Create the crew
    crew = Crew(
        agents=[analyzer, bs_generator, bo_generator, exporter],
        tasks=[analysis_task, bs_generation_task, bo_generation_task, export_task]
    )

    # -=--=--=--=--=--=--=--=--=--=--=--=--=--=--=-

    response = {}
    try:
        result = crew.kickoff(inputs={"openapi_spec": openApiSpec})
        response = {
            "status": "success",
            "result": result
        }
    except Exception as e:
        response = {
            "status": "error",
            "error": {
                "code": "agent error",
                "message": "Error during conversion agent process",
                "description": str(e),
                "details": {
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            }
        }

    return json.dumps(response)
}

}

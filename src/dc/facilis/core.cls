Class dc.facilis.core Extends %RegisteredObject
{

/// Execute
ClassMethod Execute(openapi As %String, Output response) As %Status
{
        Set sc = $$$OK
        Set response = ..GenerateI14y(openapi)
        Return sc
}

/// Generate Interoperability
ClassMethod GenerateI14y(openapi As %String) As %String [ Language = python ]
{
    import os
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    from dotenv import load_dotenv
    import json
    from langchain.prompts import PromptTemplate # Import PromptTemplate

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY, verbose = True )

    def generate_iris_components(openapi_json):
        """Generates InterSystems IRIS Business Service, Business Operations, and Production from OpenAPI JSON."""
        prompt = PromptTemplate(
            input_variables=["openapi"], # Corrected input variable name
            template="""
            Convert the following OpenAPI v3 JSON into an InterSystems IRIS interoperability setup:
            
            OpenAPI JSON:
            {openapi}
            
            Generate the following:
            - A Business Service class (`CustomRESTService`) that exposes REST endpoints matching those in OpenAPI.
            - A Business Operation class for each OpenAPI endpoint.
            - A Production class that includes the Business Service and all Business Operations.
            
            Ensure that:
            - The Business Service uses `EnsLib.HTTP.InboundAdapter`.
            - Each Business Operation uses `EnsLib.NullAdapter`.
            - The Business Service routes requests to the appropriate Business Operation.
            - Each Business Operation processes the request and returns a response.
            
            Return only the ObjectScript code, with separate sections for each class.
            """
        )
        
        formatted_prompt = prompt.format(
            openapi=json.dumps(openapi_json, indent=2) # Corrected variable name
        )
        
        response = llm.invoke(formatted_prompt)
        return response
    
    openapi_json = load_json(openapi)

    iris_code_output = generate_iris_components(openapi_json)
    
    # TODO: parse markdown to cls and import as UDL
    print(iris_code_output.content)
    print("InterSystems IRIS XML generated successfully.")
    return ""
}

}

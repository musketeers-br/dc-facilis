from crewai.tools import BaseTool
# from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun

# class SearchToolSchema(BaseModel):
#     """Input for ScrapeWebsiteTool."""

#     query: str = Field(..., description="Mandatory query text to search")


class SearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = """Useful to search the internet about a given topic and return relevant results. Way better than Google..."""
    # args_schema: Type[BaseModel] = SearchToolSchema

    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)
from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# from langchain_openai import ChatOpenAI
from crewai import LLM
from crewai_tools import ScrapeWebsiteTool
from tools.search_tool import SearchTool

# # use langhcain chatllm
# llm_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)

llm_model = LLM(
    model="openai/gpt-4o-mini", # call model by provider/model_name
    temperature=0.4,
    seed=42
)

# web_search_tool = WebsiteSearchTool()
# seper_dev_tool = SerperDevTool()
# file_read_tool = FileReadTool(
#     file_path='job_description_example.md',
#     description='A tool to read the job description example file.'
# )

@CrewBase
class TripAdvisorCrew:
    """TripAdvisor crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def local_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['local_expert'],
            llm=llm_model,
            tools=[
                SearchTool(),
                ScrapeWebsiteTool(website_url="https://en.wikipedia.org/wiki/Artificial_intelligence")
            ],
            verbose=True,
        )

    @agent
    def travel_concierge(self) -> Agent:
        return Agent(
            config=self.agents_config['travel_concierge'],
            llm=llm_model,
            tools=[],
            verbose=True,
        )
    
    @task
    def gather_task(self) -> Task:
        return Task(
            config=self.tasks_config['gather_task'],
            agent=self.local_expert()
        )
    
    @task
    def plan_task(self) -> Task:
        return Task(
            config=self.tasks_config['plan_task'],
            agent=self.travel_concierge()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TripAdvisorCrew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
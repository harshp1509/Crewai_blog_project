from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List



#define the class for our crew
@CrewBase
class ResearchAndBlogCrew():
    agents: list[BaseAgent]
    tasks: list[Task]

    #define the paths of config files
    agents_config= "config/agents.yaml"
    tasks_config= "config/tasks.yaml"

    def __init__(self):
        self.groq_llm = LLM(model="groq/llama-3.1-8b-instant")

    #==============Agents==============
    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generator"],
            llm=self.groq_llm
        )
    
    @agent
    def blog_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["blog_writer"],
            llm=self.groq_llm
        )
    
    #============tasks================
    #order of task Definition matters
    @task
    def report_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_task"]
        )

    @task
    def blog_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["blog_writing_task"],
            output_file="blogs/blog.md"
        )
    
    #==================Crew=====================
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import logging
from crewai_tools import SerperDevTool

logger = logging.getLogger(__name__)

@CrewBase
class CausalCopilotAgent():
    """CausalCopilotAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config: dict
    tasks_config: dict

    # Define agents
    @agent
    def verification_agent(self) -> Agent:
        """Creates the verification agent."""
        logger.info("Initializing verification agent.")
        return Agent(config=self.agents_config['verification_agent'], verbose=True, tools=[SerperDevTool])

    # Define tasks
    @task
    def verification_task(self) -> Task:
        """Defines the verification task."""
        logger.info("Initializing verification task.")
        return Task(config=self.tasks_config['verification_task'])

    # Define the crew
    @crew
    def crew(self) -> Crew:
        """Creates the CausalCopilotAgent crew."""
        logger.info("Creating the crew with agents and tasks.")
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

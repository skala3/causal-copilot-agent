from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import logging

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
    def planner_agent(self) -> Agent:
        """Creates the planner agent."""
        logger.info("Initializing planner agent.")
        return Agent(config=self.agents_config['planner_agent'], verbose=True)

    @agent
    def data_agent(self) -> Agent:
        """Creates the data preprocessing agent."""
        logger.info("Initializing data agent.")
        return Agent(config=self.agents_config['data_agent'], verbose=True)

    @agent
    def structure_discovery_agent(self) -> Agent:
        """Creates the structure discovery agent."""
        logger.info("Initializing structure discovery agent.")
        return Agent(config=self.agents_config['structure_discovery_agent'], verbose=True)

    @agent
    def scm_builder_agent(self) -> Agent:
        """Creates the SCM builder agent."""
        logger.info("Initializing SCM builder agent.")
        return Agent(config=self.agents_config['scm_builder_agent'], verbose=True)

    @agent
    def model_validator_agent(self) -> Agent:
        """Creates the SCM evaluation agent."""
        logger.info("Initializing model validator agent.")
        return Agent(config=self.agents_config['model_validator_agent'], verbose=True)

    @agent
    def model_memory_agent(self) -> Agent:
        """Creates the causal model memory manager agent."""
        logger.info("Initializing model memory agent.")
        return Agent(config=self.agents_config['model_memory_agent'], verbose=True)

    @agent
    def intervention_agent(self) -> Agent:
        """Creates the intervention simulation agent."""
        logger.info("Initializing intervention agent.")
        return Agent(config=self.agents_config['intervention_agent'], verbose=True)

    @agent
    def counterfactual_agent(self) -> Agent:
        """Creates the counterfactual analysis agent."""
        logger.info("Initializing counterfactual agent.")
        return Agent(config=self.agents_config['counterfactual_agent'], verbose=True)

    @agent
    def simulation_agent(self) -> Agent:
        """Creates the forward simulation agent."""
        logger.info("Initializing simulation agent.")
        return Agent(config=self.agents_config['simulation_agent'], verbose=True)

    @agent
    def reporting_agent(self) -> Agent:
        """Creates the reporting agent."""
        logger.info("Initializing reporting agent.")
        return Agent(config=self.agents_config['reporting_agent'], verbose=True)

    # Define tasks
    @task
    def planning_task(self) -> Task:
        """Defines the planning task."""
        logger.info("Initializing planning task.")
        return Task(config=self.tasks_config['planning_task'])

    @task
    def data_preprocessing_task(self) -> Task:
        """Defines the data preprocessing task."""
        logger.info("Initializing data preprocessing task.")
        return Task(config=self.tasks_config['data_preprocessing_task'])

    @task
    def structure_learning_task(self) -> Task:
        """Defines the structure learning task."""
        logger.info("Initializing structure learning task.")
        return Task(config=self.tasks_config['structure_learning_task'])

    @task
    def scm_construction_task(self) -> Task:
        """Defines the SCM construction task."""
        logger.info("Initializing SCM construction task.")
        return Task(config=self.tasks_config['scm_construction_task'])

    @task
    def intervention_simulation_task(self) -> Task:
        """Defines the intervention simulation task."""
        logger.info("Initializing intervention simulation task.")
        return Task(config=self.tasks_config['intervention_simulation_task'])

    @task
    def counterfactual_analysis_task(self) -> Task:
        """Defines the counterfactual analysis task."""
        logger.info("Initializing counterfactual analysis task.")
        return Task(config=self.tasks_config['counterfactual_analysis_task'])

    @task
    def reporting_task(self) -> Task:
        """Defines the reporting task."""
        logger.info("Initializing reporting task.")
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file=self.tasks_config['reporting_task'].get('output_file', 'report.md')
        )

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

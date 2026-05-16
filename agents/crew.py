from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set the OpenAI API key    

my_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def build_crew(project_description):

    planner = Agent(
        role="Planner",
        goal="Generate structured project tasks",
        backstory="Expert in WBS",
        llm="gpt-4o-mini"
    )

    estimator = Agent(
        role="Estimator",
        goal="Estimate duration, cost, resources",
        backstory="Expert in project estimation",
        llm="gpt-4o-mini"
       
    )

    dependency_agent = Agent(
        role="Dependency Manager",
        goal="Assign valid dependencies",
        backstory="Expert in project scheduling and dependencies",
        llm="gpt-4o-mini"
    )

    t1 = Task(
        description=f"""
        Break into tasks, leave as empty duration, cost, resources, and dependencies.

        OUTPUT STRICT JSON:
        {{
          "tasks":[
            {{"id":"T1","name":"","duration":0,"resources":[],"cost":0,"dependencies":[]}}
          ]
        }}
        Project: {project_description}
        """,
        expected_output="JSON with tasks, empty duration, cost, resources, dependencies",
        agent=planner
    )

    t2 = Task(
        description="Fill duration, cost, resources properly. Keep JSON format.",
        expected_output="JSON with tasks, filled duration, cost, resources, dependencies",
        context=[t1],
        agent=estimator
    )

    t3 = Task(
        description="Add valid dependencies (no cycles). Keep JSON format.",
        expected_output="JSON with tasks, filled duration, cost, resources, dependencies. JSON must be correctly formatted with no added text.",
        context=[t2],
        agent=dependency_agent
    )

    crew = Crew(
        agents=[planner, estimator, dependency_agent],
        tasks=[t1, t2, t3],
        verbose=True
    )

    return crew.kickoff()
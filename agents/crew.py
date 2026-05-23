from crewai import Agent, Task, Crew, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
   

my_llm = LLM(
     model ="gemini/gemini-2.5-flash",
    api_key=st.secrets["GEMINI"]
)


def build_crew(project_description):

    planner = Agent(
        role="Planner",
        goal="Generate structured project tasks",
        backstory="Expert in WBS",
        llm=my_llm
    )

    estimator = Agent(
        role="Estimator",
        goal="Estimate duration, cost, resources",
        backstory="Expert in project estimation",
        llm=my_llm
       
    )

    dependency_agent = Agent(
        role="Dependency Manager",
        goal="Assign valid dependencies",
        backstory="Expert in project scheduling and dependencies",
        llm=my_llm
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
        expected_output="JSON with tasks, filled duration, cost, resources, dependencies. JSON must be correctly formatted with no added text.resources should be a list of strings (e.g. [string1, string2])",",
        context=[t2],
        agent=dependency_agent
    )

    crew = Crew(
        agents=[planner, estimator, dependency_agent],
        tasks=[t1, t2, t3],
        verbose=True
    )

    return crew.kickoff()

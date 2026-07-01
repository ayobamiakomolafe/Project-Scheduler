from crewai import Agent, Task, Crew, LLM
from core.models import ProjectWBS
import os


def build_crew(project_description, llm, rag_tool=None):
    """
    Builds and runs the Planner -> Estimator -> Dependency Manager crew.

    Args:
        project_description: Free-text description of the project.
        llm: A crewai LLM instance (selected by the user in the UI).
        rag_tool: Optional CrewAI tool for looking up real salary/role data
                  from an uploaded company markdown file. If None, the
                  Estimator falls back to a flat $400/day blended rate.
    """
    # --- AGENTS ---
    planner = Agent(
        role="Planner",
        goal="Deconstruct project descriptions into a highly structured Work Breakdown Structure (WBS).",
        backstory="Expert Project Management Officer skilled in WBS creation.",
        llm=llm
    )

    estimator_tools = [rag_tool] if rag_tool else []

    if rag_tool:
        estimator_backstory = (
            "An expert actuary. Use the tool to find real internal salary and roles "
            "rates for each role. If a role requested by the Planner is not found exactly "
            "in the database, look for the closest matching seniority tier (Junior, Mid, "
            "Senior)."
        )
    else:
        estimator_backstory = (
            "An expert actuary and project manager that estimates right personnel roles for tasks, task durations in working days, and compute salary costs. "
    
        )

    estimator = Agent(
        role="Cost and Resource Estimator",
        goal="Assign accurate personnel roles, task durations in working days, and compute costs.",
        backstory=estimator_backstory,
        tools=estimator_tools,
        llm=llm
    )

    dependency_manager = Agent(
        role="Dependency Manager",
        goal="Establish logical, acyclic execution pathways between tasks.",
        backstory="A scheduling expert specialized in Critical Path Method (CPM) and network diagrams.",
        llm=llm
    )

    # --- TASKS ---
    t1 = Task(
        description=f"Analyze this project description and break it into structural tasks: {project_description}",
        expected_output="A structured list of tasks with names and IDs.",
        output_json=ProjectWBS,
        agent=planner
    )

    t2_description = (
        "Review the generated tasks. Match experience levels to task complexity, "
        + ("use the RAG tool to look up and assign real company roles and extract their salary rates, "
           if rag_tool else
           "compute the salary and total task cost based on estimated duration.")
         
    )

    t2 = Task(
        description=t2_description,
        expected_output="WBS data fully populated with realistic duration, required resource titles, and calculated costs.",
        output_json=ProjectWBS,
        context=[t1],
        agent=estimator
    )

    t3 = Task(
        description="Review the populated project data and inject logical dependencies. Ensure there are no cyclical dependencies (e.g., T1 depends on T2, which depends on T1).",
        expected_output="Final clean project layout including verified acyclic dependencies.",
        output_json=ProjectWBS,
        context=[t2],
        agent=dependency_manager
    )

    crew = Crew(
        agents=[planner, estimator, dependency_manager],
        tasks=[t1, t2, t3],
        verbose=False
    )

    return crew.kickoff()

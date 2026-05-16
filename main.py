from fastapi import FastAPI
from agents.crew import build_crew
from utils.parser import safe_parse_json
from core.validator import validate_project
from core.scheduler import schedule_tasks
from tools.gantt_generator import generate_gantt
from tools.doc_generator import generate_doc

app = FastAPI()

@app.post("/schedule")
def generate_schedule(description: str):
    result = build_crew(description)
   
    data = safe_parse_json(result.raw)
    validated = validate_project(data)

    schedule = schedule_tasks(validated.tasks)

    generate_gantt(schedule)
    generate_doc(schedule)

    return {
        "message": "Project generated",
        "tasks": schedule
    }
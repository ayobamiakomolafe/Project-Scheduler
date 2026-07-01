import os
import json
from crewai import Agent, Crew, Task, LLM
from agents.crew import build_crew
from utils.parser import safe_parse_json
from core.models import ProjectWBS

# Pin temperature to 0.0 for academic determinism and low cost
gemini_engine = LLM(model="gemini/gemini-2.5-flash", temperature=0.0, api_key=os.getenv("GEMINI"))
gpt_engine = LLM(model="openai/gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_KEY"))
llama_engine = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.0, api_key=os.getenv("GROQ_API_KEY"))

desc_a = "(FinTech Engine): “Build a secure, regulatory-compliant healthcare patient portal with telemedicine video integration, scheduling, and Stripe payment processing.”"
desc_b = " (Data/AI Pipeline): “Develop an enterprise automated data ingestion pipeline that fetches real-time IoT sensor logs, stores them in Snowflake, and trains an anomaly detection model.”"
desc_c = "(Mobile App): “Create a cross-platform food delivery mobile application featuring real-time GPS tracking, dual interfaces (customer and driver), and push notifications.”"

models_pool = {"Gemini 2.5 Flash": gemini_engine, "GPT-4o-Mini": gpt_engine, "Llama-3-70B": llama_engine}
projects_pool = {"Project A": desc_a, "Project B": desc_b, "Project C": desc_c}



# Run 3 iterations per intersection to collect data for your tables
for model_name, engine in models_pool.items():
    for proj_name, description in projects_pool.items():
        for iteration in range(1, 2):  # Run 2 iterations per intersection
            print(f"\n--- Running {model_name} on {proj_name}, Iteration {iteration} ---")
            result = build_crew(description, engine)
            data = safe_parse_json(result.raw)
            output_name = f"{model_name.replace(' ', '_')}_{proj_name.replace(' ', '_')}_iter{iteration}.json"
            with open(output_name, "w") as f:
                json.dump(data, f, indent=4)



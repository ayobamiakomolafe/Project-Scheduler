import os
import json
import time
from google import genai
from google.genai import types
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# 1. INITIALIZE CLIENTS & CONFIGURATION
# =====================================================================
# Ensure your environment variables are set: 
# GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY

gemini_client = genai.Client(api_key=os.getenv("GEMINI"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Global execution parameters
NUM_ITERATIONS = 2
GLOBAL_TEMPERATURE = 0.0

# =====================================================================
# 2. DEFINE PROJECT CATALOG INPUTS
# =====================================================================

desc_a = "(FinTech Engine): “Build a secure, regulatory-compliant healthcare patient portal with telemedicine video integration, scheduling, and Stripe payment processing.”"
desc_b = " (Data/AI Pipeline): “Develop an enterprise automated data ingestion pipeline that fetches real-time IoT sensor logs, stores them in Snowflake, and trains an anomaly detection model.”"
desc_c = "(Mobile App): “Create a cross-platform food delivery mobile application featuring real-time GPS tracking, dual interfaces (customer and driver), and push notifications.”"

PROJECTS = {
    "Project_A": desc_a,
    "Project_B": desc_b,
    "Project_C": desc_c
}

# =====================================================================
# 3. SYSTEM PROMPT & OUTPUT EXPECTATION
# =====================================================================
VANILLA_SYSTEM_PROMPT = """You are an expert technical project manager and software architect. 
Your task is to generate a comprehensive Work Breakdown Structure (WBS), schedule estimation, and costing model based on the provided project description.

You must output your response STRICTLY as a single, valid JSON object matching the schema below. 
Do not include markdown code block formatting (like ```json), introduction text, or conversational explanations. Just the raw JSON.

EXPECTED JSON SCHEMA:
{
  "tasks": [
    {
      "id": "TASK-001",
      "name": "Task or Phase Title",
      "roles": ["Assigned Engineering Roles"],
      "duration in working days": 5.0,
      "cost in USD": 4500.00,
      "dependencies": []
    }
  ]
}

CRITICAL RULES:
1. For 'parent_id': Use 'ROOT' for top-level project phases. If a task belongs inside a phase, map its 'parent_id' to that phase's task 'id' to build a tree hierarchy.
2. For 'duration': Provide estimates in decimal days.
3. For 'cost': Provide a realistic, flat-rate financial estimate for the task.
4. For 'dependencies': List the 'id' strings of any tasks that must finish before this task can start. Avoid any circular logic.

Company data to select roles and salary/cost values

## 1. System Engineering & Architecture Tiers

Highly critical infrastructure, security auditing, and global database schema layouts must utilize resources strictly from this tier to ensure structural integrity and regulatory compliance.

### Principal Cloud Architect
*   **Resource Code:** ARCH-PRIN-01
*   **Daily Billable Rate:** $175.00
*   **Target Domains:** Global FinTech systems, highly scaled Cloud Infrastructures, Complex Microservices, IoT Broker Gateways.
*   **Mandatory Domain Keywords:** architect, infrastructure, snowflake, infrastructure provisioning, compliance frameworks.

### Senior Software Architect
*   **Resource Code:** ARCH-SR-02
*   **Daily Billable Rate:** $145.00
*   **Target Domains:** Multi-tenant Portals, Database Sharding, Payment Ingestion Systems, Real-Time Ingestion Topologies.
*   **Mandatory Domain Keywords:** schema layout, database architect, security architecture, stripe integrations, telemetry.

---

## 2. Core Backend & Data Engineering Tiers

These roles handle functional business logic, API route execution, stream processing pipelines, and data warehouse persistence engines.

### Lead Data Engineer
*   **Resource Code:** DATA-LEAD-01
*   **Daily Billable Rate:** $130.00
*   **Target Domains:** Large-Scale Automated Data Pipelines, Snowflake Ingestion, Real-Time Sensor Stream Processing, Vector Indexes.
*   **Mandatory Domain Keywords:** snowflake pipeline, automation scripts, iot logs, anomaly detection models, analytics layer.

### Senior Backend Engineer
*   **Resource Code:** ENG-SR-BACK
*   **Daily Billable Rate:** $115.00
*   **Target Domains:** Telemedicine Streaming, Compliance Framework Handling, REST API Infrastructure, State Routing Engines.
*   **Mandatory Domain Keywords:** video streaming integration, stripe payment backend, secure patient portals, relational tables.

### Mid-Level Backend Engineer
*   **Resource Code:** ENG-MID-BACK
*   **Daily Billable Rate:** $85.00
*   **Target Domains:** CRUD Operations, Database Migrations, Event Listeners, Third-Party Wrapper Integration.
*   **Mandatory Domain Keywords:** alembic migration script, webhook endpoint, logging middleware, telemetry ingestion.

---

## 3. Frontend & Mobile Application Engineering Tiers

These roles are responsible for client-side state machine tracking, cross-platform user experiences, sensory hardware polling, and localized real-time rendering.

### Senior Mobile Applications Engineer
*   **Resource Code:** ENG-SR-MOB
*   **Daily Billable Rate:** $110.00
*   **Target Domains:** Cross-Platform Native Environments, Real-Time Hardware Polling, Multi-State App Synchronization.
*   **Mandatory Domain Keywords:** mobile delivery application, gps tracking system, client state machine, push notifications, local storage caching.

### Frontend Developer (Mid-Level)
*   **Resource Code:** ENG-MID-FRONT
*   **Daily Billable Rate:** $80.00
*   **Target Domains:** Web Interface Rendering, Form State Validation, Component Libraries, Interactive Dashboards.
*   **Mandatory Domain Keywords:** patient portal ui, dashboard rendering, payment gateway validation, cross-browser styling.

---

## 4. Product Quality, Security & Operations (DevOps) Tiers

These roles ensure systemic throughput stability, continuous lifecycle deployment validation, automated test pipelines, and security remediation.

### Senior DevOps & Security Engineer
*   **Resource Code:** OPS-SR-SEC
*   **Daily Billable Rate:** $125.00
*   **Target Domains:** Secure CI/CD Pipelines, Static Application Security Testing (SAST), Container Orchestration, Encryption at Rest.
*   **Mandatory Domain Keywords:** cryptography, compliance auditing, telemetry deployment, containerization, identity management.

### Automated QA Engineer
*   **Resource Code:** QA-AUTO-03
*   **Daily Billable Rate:** $75.00
*   **Target Domains:** Regression Testing, End-to-End User Flow Emulation, Mock Payload Testing, Verification Scripts.
*   **Mandatory Domain Keywords:** automated test scripts, mock payload suites, boundary testing, end to end validation.
"""

# =====================================================================
# 4. LLM CALL ROUTINES
# =====================================================================
def call_gemini(prompt):
    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=VANILLA_SYSTEM_PROMPT,
            temperature=GLOBAL_TEMPERATURE,
            response_mime_type="application/json" # Enforces JSON mode at API level
        )
    )
    return response.text

def call_openai(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": VANILLA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=GLOBAL_TEMPERATURE,
        response_format={"type": "json_object"} # Enforces JSON mode
    )
    return response.choices[0].message.content

def call_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": VANILLA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=GLOBAL_TEMPERATURE,
        response_format={"type": "json_object"} # Enforces JSON mode
    )
    return response.choices[0].message.content

# Mapping matrix engines
MODEL_ROUTINES = {
    "gemini-2.5-flash": call_gemini,
    "gpt-4o-mini": call_openai,
    "llama-3-70b": call_groq
}

# =====================================================================
# 5. EXECUTION MATRIX RUNNER
# =====================================================================
def run_vanilla_matrix():
    print("🚀 Initializing Vanilla Baseline Testing Matrix Loops...")
    os.makedirs("vanilla_outputs", exist_ok=True)
    
    for model_name, call_function in MODEL_ROUTINES.items():
        print(f"\n🔄 Evaluating Engine Backend: {model_name}")
        
        for proj_key, proj_desc in PROJECTS.items():
            print(f"  📂 Processing Target Domain Block: {proj_key}")
            
            for iteration in range(1, NUM_ITERATIONS + 1):
                print(f"    [Run {iteration}/{NUM_ITERATIONS}] Generating raw unstructured dump...")
                
                output_filename = f"vanilla_outputs/{model_name}_{proj_key}_run{iteration}.json"
                
                # Construct clean prompt wrapping the project text
                user_prompt = f"Generate the structured planning output for this project:\n\n{proj_desc}"
                
                try:
                    start_time = time.time()
                    raw_result = call_function(user_prompt)
                    latency = time.time() - start_time
                    
                    # Save exact raw string to evaluate parsing vulnerability in evaluation scripts
                    with open(output_filename, "w", encoding="utf-8") as f:
                        f.write(raw_result.strip())
                        
                    print(f"      ✅ Saved output. Latency: {latency:.2f}s")
                    
                except Exception as e:
                    print(f"      ❌ Execution Error caught: {str(e)}")
                    # Save fallback broken mock text to test parsing resilience
                    with open(output_filename, "w", encoding="utf-8") as f:
                        f.write(f"CRASH_ERROR: {str(e)}")
                
                # Polite cooling delay to avoid aggressive rate limits on public endpoints
                time.sleep(2)

    print("\n🎉 Vanilla Baseline Execution Matrix Complete! Outputs stored in /vanilla_outputs.")

if __name__ == "__main__":
    run_vanilla_matrix()
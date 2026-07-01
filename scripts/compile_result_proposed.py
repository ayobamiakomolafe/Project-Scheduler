import os
import json
import statistics
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util

# Load the local NLP model for semantic quality grading
print("Loading semantic evaluation transformer model...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your Ground Truth anchors (derived from your human experts in Step 2)
# Replace these placeholder values with your actual human consensus data
HUMAN_BASELINE_DATA = {
  "Project_A": {
    "cost": 37765.0,
    "duration_mean": 10.31578947368421,
    "duration_var": 23.783625730994153,
    "depth": 10,
    "breadth": 5,
    "balance": 18,
    "names": "Project Initiation & Requirements Gathering|Regulatory Compliance & Security Planning|System Architecture & Database Schema Design|Infrastructure Provisioning & CI/CD Setup|Backend API Development for Patient Portal|Patient Authentication & Identity Management|Telemedicine Video Integration|Appointment Scheduling Module|Stripe Payment Integration|Frontend Patient Portal UI Development|Dashboard & Form Validation Implementation|Mobile Responsiveness & Cross-Browser Optimization|Database Migration & Logging Middleware Setup|Security Hardening & Encryption Implementation|Automated Testing & QA Validation|End-to-End Compliance Testing|User Acceptance Testing (UAT)|Production Deployment & Monitoring Setup|Post-Deployment Support & Optimization",
    "role_accuracy": 24.274658529382002
  },
  "Project_B": {
    "cost": 48755.0,
    "duration_mean": 10.68421052631579,
    "duration_var": 23.783625730994153,
    "depth": 18,
    "breadth": 2,
    "balance": 50,
    "names": "Project Initiation & Requirement Analysis|System Architecture & Snowflake Schema Design|Infrastructure Provisioning and Cloud|IoT Sensor Data Source Integration|Real-Time Data Ingestion Pipeline Development|Stream Processing & Data Transformation Layer|Snowflake Data Warehouse Configuration|Database Migration & Logging Middleware Setup|Anomaly Detection Model Design|Machine Learning Model Training & Validation|Automation Scripts & Scheduling Configuration|API Development for Analytics Access|Security Hardening & Compliance Auditing|CI/CD Pipeline & Containerization Setup|Automated Testing & Mock Payload Validation|End-to-End Data Flow Validation|User Acceptance Testing (UAT)|Production Deployment & Monitoring Setup|Post-Deployment Monitoring & Optimisation",
    "role_accuracy": 24.525618317880127
  },
  "Project_C": {
    "cost": 47915.0,
    "duration_mean": 10.85,
    "duration_var": 23.08157894736842,
    "depth": 12,
    "breadth": 3,
    "balance": 8,
    "names": "Project Initiation & Requirements Gathering|System Architecture & Application Design|Infrastructure Provisioning & Cloud Setup|Backend API Development for Food Ordering System|Database Schema & Migration Setup|Customer Mobile App UI/UX Development|Driver Mobile App UI/UX Development|Real-Time GPS Tracking Integration|Push Notification System Implementation|User Authentication & Identity Management|Payment & Checkout Workflow Integration|Order Management & Delivery Workflow Logic|Real-Time Status Synchronization|Security Hardening & Compliance Auditing|CI/CD Pipeline & Containerization Setup|Automated Testing & Mobile QA Validation|End-to-End Functional Testing|User Acceptance Testing (UAT)|Production Deployment & Monitoring Setup|Post-Deployment Monitoring & Optimization",
    "role_accuracy": 27.9665862955153
  }
}




def compute_role_accuracy(data, similarity_model):
    role_scores = []

    for task in data["tasks"]:
        task_text = task.get("name", "")
        role_code = task.get("resources", "")

        if not task_text or not role_code:
            continue

        # ---- Get role title ----
        role_title = "|".join(role_code)

        # ---- Semantic comparison ----
        task_emb = similarity_model.encode(task_text, convert_to_tensor=True)
        role_emb = similarity_model.encode(role_title, convert_to_tensor=True)

        score = util.pytorch_cos_sim(task_emb, role_emb).item()

        # normalize to 0–100
        score = max(score, 0.0) * 100

        role_scores.append(score)

    return np.mean(role_scores) if role_scores else np.nan

def parse_and_grade_file(filepath, project_key):
    """Parses a single JSON file and calculates all framework metrics."""
    anchor = HUMAN_BASELINE_DATA[project_key]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read().strip()

    # ---- METRIC 1: SYNTAX INTEGRITY ----
    try:
        data = json.loads(raw_text)
        syntax_success = 1.0
    except Exception:
        # Penalties assigned for broken structural files
        return {
    "syntax_success": 0.0,
    "cost_variance": np.nan,
    "loop_rate": np.nan,
    "semantic_similarity": np.nan,
    "role_accuracy": np.nan,
    "max_depth": np.nan,
    "max_breadth": np.nan,
    "tree_balance": np.nan,
    "avg_dur": np.nan,
    "dur_var": np.nan
}

    # ---- METRIC 2: COST VARIANCE ----
    total_agent_cost = 0.0
    for task in data['tasks']:
        try:
            total_agent_cost += float(task['cost'])
        except Exception:
            continue
    try:
        cost_variance = ((total_agent_cost - anchor["cost"]) / anchor["cost"]) * 100
    except Exception:
        cost_variance = np.nan

    # ---- METRIC 3: SCHEDULING CYCLES ----
    DepGraph = nx.DiGraph()
    for task in data['tasks']:
        DepGraph.add_node(task['id'])
        for dep in task.get('dependencies', []):
            DepGraph.add_edge(dep, task['id'])
    loop_rate = 1.0 if not nx.is_directed_acyclic_graph(DepGraph) else 0.0

    # ---- METRIC 4: SEMANTIC ACCURACY ----
    agent_text = "|".join([t['name'] for t in data['tasks']])
    emb_agent = similarity_model.encode(agent_text, convert_to_tensor=True)
    emb_human = similarity_model.encode(anchor["names"], convert_to_tensor=True)
    semantic_similarity = float(util.pytorch_cos_sim(emb_agent, emb_human)[0][0]) * 100

    # ---- METRIC 5: ROLE ALIGNMENT ACCURACY ----
    role_accuracy = compute_role_accuracy(data, similarity_model)

    # ---- METRIC 6: TREE TOPOLOGY ----
    tasks = data["tasks"]
    # =========================================================
    # BUILD DEPENDENCY GRAPH (DAG)
    # =========================================================
    G = nx.DiGraph()
    G.add_node("ROOT")

    for task in tasks:
        G.add_node(task["id"])

    for task in tasks:
        deps = task.get("dependencies", [])
        if deps:
            for dep in deps:
                G.add_edge(dep, task["id"])
        else:
            G.add_edge("ROOT", task["id"])

    # =========================================================
    # DEPTH (LONGEST PATH FROM ROOT)
    # =========================================================
    try:
        depths = nx.single_source_shortest_path_length(G, "ROOT")
        max_depth = max(depths.values()) if depths else 0
    except:
        max_depth = 0

    # =========================================================
    # BREADTH (MAX OUT-DEGREE = MAX PARALLEL FAN-OUT)
    # =========================================================
    max_breadth = max((G.out_degree(n) for n in G.nodes), default=0)

    # =========================================================
    # BALANCE (VARIANCE OF LEAF DEPTHS)
    # =========================================================
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]

    try:
        leaf_depths = [
            nx.shortest_path_length(G, "ROOT", leaf)
            for leaf in leaves
            if nx.has_path(G, "ROOT", leaf)
        ]

        balance = (
            statistics.variance(leaf_depths)
            if len(leaf_depths) > 1
            else 0.0
        )
    except:
        balance = 0.0

    # ---- METRIC 7: PACING & DURATION VARIANCE ----
    durations = [float(task['duration']) for task in data['tasks']]
    avg_dur = statistics.mean(durations) if durations else 0.0
    dur_var = statistics.variance(durations) if len(durations) > 1 else 0.0

    return {
        "syntax_success": syntax_success * 100, "cost_variance": cost_variance, "loop_rate": loop_rate * 100,
        "semantic_similarity": semantic_similarity, "role_accuracy": role_accuracy, "max_depth": max_depth,
        "max_breadth": max_breadth, "tree_balance": balance, "avg_dur": avg_dur, "dur_var": dur_var
    }

# =====================================================================
# AGGREGATION ENGINE ENTRY POINT
# =====================================================================

import glob

def compile_all_results():
    # Setup your folder tracking paths
    configurations = {
       
        "Proposed Framework (Gemini)": "proposed_gemini_outputs",
        "Proposed Framework (GPT)": "proposed_gpt_outputs",
        "Proposed Framework (Llama)": "proposed_llama_outputs",
    }
    
    print("\n📊 Beginning Grand Compilation of Simulation Matrix Data...")
    
    for config_label, folder in configurations.items():
        if not os.path.exists(folder):
            print(f"⚠️ Folder path '{folder}' not detected. Skipping configuration loop.")
            continue
            
        print(f"\n--- {config_label} Analysis ---")
        
        # Grab absolutely every single JSON file in this specific folder
        all_json_files = glob.glob(os.path.join(folder, "*.json"))
        
        if not all_json_files:
            print(f"   (No JSON files found inside '{folder}')")
            continue
            
        print(f"   Found {len(all_json_files)} total files to process...")

        # Macro counters for Table 1
        macro_syntax, macro_cost, macro_loops, macro_semantic, macro_role = [], [], [], [], []
        
        # Structure to group items dynamically by project for Table 2
        project_groups = {
            "Project_A": {"depths": [], "breadths": [], "balances": [], "dur_means": [], "dur_vars": []},
            "Project_B": {"depths": [], "breadths": [], "balances": [], "dur_means": [], "dur_vars": []},
            "Project_C": {"depths": [], "breadths": [], "balances": [], "dur_means": [], "dur_vars": []}
        }
        
        for filepath in all_json_files:
            filename = os.path.basename(filepath)
            
            # Smart project matching that handles "ProjectA" or "Project_A"
            matched_proj = None
            if "project_a" in filename.lower() or "projecta" in filename.lower():
                matched_proj = "Project_A"
            elif "project_b" in filename.lower() or "projectb" in filename.lower():
                matched_proj = "Project_B"
            elif "project_c" in filename.lower() or "projectc" in filename.lower():
                matched_proj = "Project_C"
                
            if not matched_proj:
                print(f"      ⚠️ Skipping file {filename} (Could not determine Project A, B, or C from name)")
                continue

            # Grade the file
            metrics = parse_and_grade_file(filepath, matched_proj)
            
            # Append directly to Table 1 metrics lists
            macro_syntax.append(metrics["syntax_success"])
            macro_cost.append(abs(metrics["cost_variance"]))
            macro_loops.append(metrics["loop_rate"])
            macro_semantic.append(metrics["semantic_similarity"])
            macro_role.append(metrics["role_accuracy"])
            
            # Append directly to Table 2 data buckets
            pg = project_groups[matched_proj]
            pg["depths"].append(metrics["max_depth"])
            pg["breadths"].append(metrics["max_breadth"])
            pg["balances"].append(metrics["tree_balance"])
            pg["dur_means"].append(metrics["avg_dur"])
            pg["dur_vars"].append(metrics["dur_var"])

        # ---- PRINT TABLE 2 DATA FOR THIS CONFIGURATION ----
        for proj, data_buckets in project_groups.items():
            if data_buckets["depths"]: # Make sure we processed at least one file for this project
                print(f"📍 {proj} Dimensions -> "
                      f"Max Depth: {np.nanmean(data_buckets['depths']):.1f} | "
                      f"Max Breadth: {np.nanmean(data_buckets['breadths']):.1f} | "
                      f"Balance Var: {np.nanmean(data_buckets['balances']):.2f} | "
                      f"Task Mean: {np.nanmean(data_buckets['dur_means']):.1f} Days | "
                      f"Pacing Var: {np.nanmean(data_buckets['dur_vars']):.2f}")
        
        # ---- PRINT TABLE 1 DATA FOR THIS CONFIGURATION ----
        if macro_syntax:
            print(f"📈 Architectural Profile Metrics Summary:")
            print(f"   Syntax Parse Success: {np.nanmean(macro_syntax):.1f}%")
            print(f"   Mean Cost Deviation: ±{np.nanmean(macro_cost):.1f}%")
            print(f"   Dependency Loop Failure Rate: {np.nanmean(macro_loops):.1f}%")
            print(f"   Semantic Overlap Accuracy: {np.nanmean(macro_semantic):.1f}%")
            print(f"   Role Assignment Accuracy: {np.nanmean(macro_role):.1f}%")
            
if __name__ == "__main__":
    compile_all_results()
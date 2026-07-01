import networkx as nx
import statistics
import json
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
# Load the local NLP model for semantic quality grading
print("Loading semantic evaluation transformer model...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_role_accuracy(data, similarity_model):
    role_scores = []

    for task in data["tasks"]:
        task_text = task.get("name", "")
        role_code = task.get("roles", "")

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

    return role_scores

def calculate_baseline_metrics(data):
    

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

  


    tasks = data["tasks"]

    # ---------------------------
    # COST
    # ---------------------------
    total_cost = 0.0
    for task in tasks:
        try:
            total_cost += float(task.get("cost in USD", 0))
        except:
            continue

    # ---------------------------
    # DURATIONS
    # ---------------------------
    durations = []
    for task in tasks:
        try:
            durations.append(float(task.get("duration in working days", 0)))
        except:
            continue

    duration_mean = statistics.mean(durations) if durations else 0.0
    duration_var = (
        statistics.variance(durations) if len(durations) > 1 else 0.0
    )
    
    tasks_name = []
    
    for task in tasks:
        tasks_name.append(task["name"])
       
    
    role_scores = compute_role_accuracy(data, similarity_model)



    # ---------------------------
    # RETURN BASELINE METRICS
    # ---------------------------
    return {
        "cost": total_cost,
        "duration_mean": duration_mean,
        "duration_var": duration_var,
        "depth": max_depth,
        "breadth":  max_breadth,
        "balance": balance,
        "names": "|".join(tasks_name),
        "role_accuracy": np.mean(role_scores) if role_scores else np.nan
    }

human_baseline = {}
base={"Project_A": "human_projecta.json", "Project_B":"human_projectb.json", "Project_C": "human_projectc.json"}
for a, b in base.items():
    fob = open(b, "r")
    data = json.load(fob)
    result = calculate_baseline_metrics(data)
    human_baseline[a] = result

with open("human_baseline.json", "w", encoding="utf-8") as f:
    json.dump(human_baseline, f, indent=2)


import streamlit as st
import pandas as pd
from datetime import datetime
import os
import hashlib
import tempfile
from crewai import LLM
from agents.crew import build_crew
from utils.parser import safe_parse_json
from core.validator import validate_project
from core.scheduler import schedule_tasks
from tools.gantt_generator import generate_gantt
from tools.doc_generator import generate_doc
from tools.embedding import build_vector_db_from_file
from tools.rag_query_tool import make_rag_tool

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Project Scheduler",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 18px; padding: 10px 20px; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .llm-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .pill-ready { background-color: #d4edda; color: #1c6b32; }
    .pill-needs-key { background-color: #fff3cd; color: #8a6d00; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LLM provider configuration
# ------------------------------------------------------------------
LLM_CONFIGS = {
    "Llama 3.3 70B (Groq)": {
        "model": "groq/llama-3.3-70b-versatile",
        "default_env": "GROQ_API_KEY",
        "key_label": "Groq API Key (optional — a shared default key is used if left blank)",
        "has_default": True,
        "get_key_url": "https://console.groq.com/keys",
    },
    "GPT-4o mini (OpenAI)": {
        "model": "openai/gpt-4o-mini",
        "default_env": None,
        "key_label": "OpenAI API Key (required)",
        "has_default": False,
        "get_key_url": "https://platform.openai.com/api-keys",
    },
    "Gemini 2.5 Flash (Google)": {
        "model": "gemini/gemini-2.5-flash",
        "default_env": None,
        "key_label": "Gemini API Key (required)",
        "has_default": False,
        "get_key_url": "https://aistudio.google.com/apikey",
    },
}


def get_llm(provider_label: str, user_api_key: str):
    """Builds a crewai LLM instance for the chosen provider, or returns an error message."""
    cfg = LLM_CONFIGS[provider_label]
    api_key = (user_api_key or "").strip()

    if not api_key:
        if cfg["has_default"]:
            api_key = os.getenv(cfg["default_env"])
            if not api_key:
                return None, (
                    "The default Groq key isn't configured on this server, and you "
                    "haven't entered your own. Please add your Groq API key in the sidebar."
                )
        else:
            return None, f"Please enter your {provider_label} API key in the sidebar to continue."

    llm = LLM(model=cfg["model"], temperature=0.0, api_key=api_key)
    return llm, None


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
for key, default in [
    ("project_data", None),
    ("schedule", None),
    ("gantt_generated", False),
    ("rag_vector_store", None),
    ("rag_file_hash", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("# 📊 Project Scheduler")
st.markdown("#### Generate optimized project schedules with AI-powered task breakdown and scheduling")

# ------------------------------------------------------------------
# Sidebar — model + RAG configuration
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 AI Model")
    provider_label = st.selectbox(
        "Choose the LLM that powers the agents",
        options=list(LLM_CONFIGS.keys()),
        index=0,  # Llama is default
        help="Llama (Groq) works out of the box. GPT and Gemini require your own API key."
    )
    cfg = LLM_CONFIGS[provider_label]

    user_api_key = st.text_input(
        cfg["key_label"],
        type="password",
        placeholder="sk-... / gsk_... / AIza...",
        key=f"api_key_{provider_label}"
    )

    if cfg["has_default"] and not user_api_key:
        st.markdown(
            '<span class="pill pill-ready">✓ Ready — using default shared key</span>',
            unsafe_allow_html=True
        )
        st.caption(
            "If the shared key runs out of quota, enter your own Groq key above to keep going."
        )
    elif user_api_key:
        st.markdown('<span class="pill pill-ready">✓ Key provided</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-needs-key">⚠ API key required</span>', unsafe_allow_html=True)
        st.caption(f"[Get a {provider_label.split('(')[0].strip()} API key]({cfg['get_key_url']})")

    st.markdown("---")
    st.markdown("## 🗂️ Salary Database (optional)")
    uploaded_md = st.file_uploader(
        "Upload a file of company roles & salary rates",
        type=["md","txt","pdf","csv","docx","xlsx","json"],
        help=(
            "If you upload a file, the Estimator agent will look up real salary rates "
            "from it. If you skip this, every task is costed using LLM rate "
            "blended rate."
        )
    )
    if uploaded_md is not None:
        st.markdown(
            '<span class="pill pill-ready">✓ Will use uploaded salary data</span>',
            unsafe_allow_html=True
        )
    else:
        st.caption("No file uploaded — costs will default to LLM per role.")

    st.markdown("---")
    st.markdown("## 📋 About")
    st.info(
        """
        **Project Scheduler** uses AI to:
        - Break down your project into tasks
        - Estimate duration, cost, and resources
        - Manage task dependencies
        - Generate Gantt charts and reports
        """
    )
    st.markdown("### 💡 Tips")
    st.markdown(
        """
        - Be specific in your project description
        - Include scope, timeline expectations, and key requirements
        - The AI will extract and structure all details automatically
        """
    )


def get_rag_tool():
    """
    Builds (and caches in session_state) a RAG tool from the uploaded file,
    or returns None if no file was uploaded. Rebuilds only when the file changes.
    """
    if uploaded_md is None:
        return None

    file_hash = hashlib.sha256(uploaded_md.getvalue()).hexdigest()

    if st.session_state.rag_file_hash == file_hash and st.session_state.rag_vector_store is not None:
        return make_rag_tool(st.session_state.rag_vector_store)

    persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_{file_hash[:12]}")
    uploaded_md.seek(0)  # reset pointer before build_vector_db_from_file reads it
    vector_store, n_chunks = build_vector_db_from_file(uploaded_md, persist_directory=persist_dir)
    st.session_state.rag_vector_store = vector_store
    st.session_state.rag_file_hash = file_hash
    st.toast(f"Indexed {n_chunks} chunks from your salary database.", icon="📚")
    return make_rag_tool(vector_store)


# ------------------------------------------------------------------
# Main tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Input", "📊 Tasks", "📈 Schedule", "📄 Report"])

# ===== TAB 1: INPUT =====
with tab1:
    st.markdown("### Create a New Project")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Project Description")
        project_description = st.text_area(
            "Describe your project in detail",
            placeholder="Example: Build a mobile app for e-commerce with user authentication, product catalog, shopping cart, payment processing, and order tracking. Timeline: 3 months, Team: 5 people, Budget: $50,000",
            height=200,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("##### Quick Tips")
        st.markdown(
            """
            Include:
            - **Project goals** and deliverables
            - **Scope** and features
            - **Timeline** expectations
            - **Team size** and skills
            - **Budget** constraints
            - **Key milestones**

            Example format:
            > *Build an event management platform with user registration, event creation, booking system, and analytics dashboard. 6-month timeline, 4-person team, $100k budget.*
            """
        )

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🚀 Generate Schedule", use_container_width=True, type="primary"):
            if not project_description.strip():
                st.error("❌ Please enter a project description")
            else:
                llm, llm_error = get_llm(provider_label, user_api_key)
                if llm_error:
                    st.error(f"❌ {llm_error}")
                else:
                    try:
                        with st.spinner("🗂️ Preparing salary database (if provided)..."):
                            rag_tool = get_rag_tool()

                        with st.spinner(f"🤖 Analyzing project with {provider_label}..."):
                            result = build_crew(project_description, llm, rag_tool)

                        with st.spinner("✅ Parsing project structure..."):
                            data = safe_parse_json(result.raw)

                        with st.spinner("✅ Validating project data..."):
                            validated = validate_project(data)

                        with st.spinner("✅ Scheduling tasks..."):
                            schedule = schedule_tasks(validated.tasks)

                        with st.spinner("✅ Generating visualizations..."):
                            generate_gantt(schedule)
                            generate_doc(schedule)

                        st.session_state.project_data = {
                            "description": project_description,
                            "tasks": validated.tasks,
                            "raw": result.raw
                        }
                        st.session_state.schedule = schedule
                        st.session_state.gantt_generated = True

                        st.markdown("""
                        <div class="success-box">
                            <strong>✨ Project Generated Successfully!</strong><br>
                            Navigate to other tabs to view tasks, schedule, and reports.
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Error generating schedule: {str(e)}")
                        st.info("Double-check your API key in the sidebar and try again.")

# ===== TAB 2: TASKS =====
with tab2:
    if st.session_state.project_data is None:
        st.info("📝 Generate a project first using the 'Input' tab")
    else:
        st.markdown("### 📋 Project Tasks")

        tasks = st.session_state.project_data['tasks']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tasks", len(tasks), delta=None)
        with col2:
            total_duration = sum(t.duration for t in tasks)
            st.metric("Total Duration (days)", total_duration)
        with col3:
            total_cost = sum(t.cost for t in tasks)
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col4:
            total_resources = len(set(r for t in tasks for r in t.resources))
            st.metric("Unique Resources", total_resources)

        st.divider()

        st.markdown("#### Task Details")

        task_data = []
        for task in tasks:
            task_data.append({
                "ID": task.id,
                "Task Name": task.name,
                "Duration (days)": task.duration,
                "Cost": f"${task.cost:,.2f}",
                "Resources": ", ".join(task.resources) if task.resources else "N/A",
                "Dependencies": ", ".join(task.dependencies) if task.dependencies else "None"
            })

        df = pd.DataFrame(task_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Tasks as CSV",
            data=csv,
            file_name=f"tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ===== TAB 3: SCHEDULE =====
with tab3:
    if st.session_state.schedule is None:
        st.info("📝 Generate a project first using the 'Input' tab")
    else:
        st.markdown("### 📈 Project Schedule & Gantt Chart")

        schedule = st.session_state.schedule

        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = min(s['start'] for s in schedule)
            st.metric("Project Start", start_date.strftime("%Y-%m-%d"))
        with col2:
            end_date = max(s['end'] for s in schedule)
            st.metric("Project End", end_date.strftime("%Y-%m-%d"))
        with col3:
            total_days = (max(s['end'] for s in schedule) - min(s['start'] for s in schedule)).days
            st.metric("Project Duration", f"{total_days} days")

        st.divider()

        st.markdown("#### Gantt Chart Visualization")

        if os.path.exists("gantt.png"):
            from PIL import Image
            img = Image.open("gantt.png")
            st.image(img, use_container_width=True)

            with open("gantt.png", "rb") as f:
                st.download_button(
                    label="📥 Download Gantt Chart",
                    data=f.read(),
                    file_name=f"gantt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        else:
            st.warning("Gantt chart not found. Please regenerate the project.")

        st.divider()

        st.markdown("#### Detailed Schedule")

        schedule_data = []
        for item in schedule:
            schedule_data.append({
                "Task": item["task"],
                "Start Date": item["start"].strftime("%Y-%m-%d"),
                "End Date": item["end"].strftime("%Y-%m-%d"),
                "Duration (days)": item["duration"],
                "Cost": f"${item['cost']:,.2f}",
                "Resources": ", ".join(item["resources"]) if item["resources"] else "N/A",
                "Dependencies": ", ".join(item["dependencies"]) if item["dependencies"] else "None"
            })

        df_schedule = pd.DataFrame(schedule_data)
        st.dataframe(df_schedule, use_container_width=True, hide_index=True)

        csv_schedule = df_schedule.to_csv(index=False)
        st.download_button(
            label="📥 Download Schedule as CSV",
            data=csv_schedule,
            file_name=f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ===== TAB 4: REPORT =====
with tab4:
    if st.session_state.schedule is None:
        st.info("📝 Generate a project first using the 'Input' tab")
    else:
        st.markdown("### 📄 Project Report")

        st.markdown("#### Generated Documents")

        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("project.docx"):
                st.markdown("##### 📋 Project Report (Word)")
                st.info("A detailed project schedule report in Microsoft Word format containing all tasks, timeline, and resource allocation.")

                with open("project.docx", "rb") as f:
                    st.download_button(
                        label="📥 Download Project Report (DOCX)",
                        data=f.read(),
                        file_name=f"project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
            else:
                st.warning("Project report not found. Please regenerate the project.")

        with col2:
            if os.path.exists("gantt.png"):
                st.markdown("##### 📈 Gantt Chart (PNG)")
                st.info("Visual timeline representation of all tasks and their dependencies.")

                with open("gantt.png", "rb") as f:
                    st.download_button(
                        label="📥 Download Gantt Chart (PNG)",
                        data=f.read(),
                        file_name=f"gantt_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.warning("Gantt chart not found. Please regenerate the project.")

        st.divider()

        st.markdown("#### Project Summary")

        schedule = st.session_state.schedule
        project_data = st.session_state.project_data

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📋 Tasks</h4>
                <h2>{}</h2>
            </div>
            """.format(len(schedule)), unsafe_allow_html=True)

        with summary_col2:
            total_cost = sum(item["cost"] for item in schedule)
            st.markdown("""
            <div class="metric-card">
                <h4>💰 Total Cost</h4>
                <h2>${:,.2f}</h2>
            </div>
            """.format(total_cost), unsafe_allow_html=True)

        with summary_col3:
            total_duration = sum(item["duration"] for item in schedule)
            st.markdown("""
            <div class="metric-card">
                <h4>⏱️ Total Duration</h4>
                <h2>{} days</h2>
            </div>
            """.format(total_duration), unsafe_allow_html=True)

        st.divider()

        st.markdown("#### Project Description")
        st.write(project_data['description'])

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 2rem;'>
    <p>🚀 Project Scheduler v1.1 | Powered by CrewAI & Streamlit</p>
</div>
""", unsafe_allow_html=True)

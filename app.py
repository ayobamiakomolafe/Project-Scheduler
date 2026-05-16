import streamlit as st
import pandas as pd
from datetime import datetime
import os
from agents.crew import build_crew
from utils.parser import safe_parse_json
from core.validator import validate_project
from core.scheduler import schedule_tasks
from tools.gantt_generator import generate_gantt
from tools.doc_generator import generate_doc
import base64

# Page configuration
st.set_page_config(
    page_title="Project Scheduler",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        padding: 10px 20px;
    }
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'project_data' not in st.session_state:
    st.session_state.project_data = None
if 'schedule' not in st.session_state:
    st.session_state.schedule = None
if 'gantt_generated' not in st.session_state:
    st.session_state.gantt_generated = False

# Header
st.markdown("# 📊 Project Scheduler")
st.markdown("#### Generate optimized project schedules with AI-powered task breakdown and scheduling")

# Sidebar
with st.sidebar:
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
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown(
        """
        - Be specific in your project description
        - Include scope, timeline expectations, and key requirements
        - The AI will extract and structure all details automatically
        """
    )

# Main tabs
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
                try:
                    with st.spinner("🤖 Analyzing project with AI..."):
                        # Step 1: Build crew and get raw data
                        result = build_crew(project_description)
                    
                    with st.spinner("✅ Parsing project structure..."):
                        # Step 2: Parse JSON
                        data = safe_parse_json(result.raw)
                    
                    with st.spinner("✅ Validating project data..."):
                        # Step 3: Validate
                        validated = validate_project(data)
                    
                    with st.spinner("✅ Scheduling tasks..."):
                        # Step 4: Schedule
                        schedule = schedule_tasks(validated.tasks)
                    
                    with st.spinner("✅ Generating visualizations..."):
                        # Step 5: Generate artifacts
                        generate_gantt(schedule)
                        generate_doc(schedule)
                    
                    # Store in session state
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
                    st.info("Make sure you have OPENAI_API_KEY set in your .env file")

# ===== TAB 2: TASKS =====
with tab2:
    if st.session_state.project_data is None:
        st.info("📝 Generate a project first using the 'Input' tab")
    else:
        st.markdown("### 📋 Project Tasks")
        
        tasks = st.session_state.project_data['tasks']
        
        # Summary metrics
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
        
        # Tasks table
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
        
        # Download as CSV
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
        
        # Schedule metrics
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
        
        # Gantt Chart
        st.markdown("#### Gantt Chart Visualization")
        
        if os.path.exists("gantt.png"):
            from PIL import Image
            img = Image.open("gantt.png")
            st.image(img, use_container_width=True)
            
            # Download gantt chart
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
        
        # Schedule table
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
        
        # Download schedule as CSV
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
        
        # Word document
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
        
        # Gantt chart
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

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 2rem;'>
    <p>🚀 Project Scheduler v1.0 | Powered by CrewAI & Streamlit</p>
</div>
""", unsafe_allow_html=True)

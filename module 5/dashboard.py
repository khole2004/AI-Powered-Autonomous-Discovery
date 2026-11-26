import json
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Research Synthesis Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Load synthesis output
folder = Path(__file__).parent
input_file = folder / "synthesis_output.json"

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Dashboard title
st.title("🔬 Research Synthesis Dashboard")
st.write("AI-Powered Autonomous Discovery System")

st.divider()

# Paper details
st.header("Research Paper Overview")

st.metric("Paper ID", data.get("paper_id", "Unknown"))

summary = data.get("research_summary", {})

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Images", summary.get("total_images", 0))

with col2:
    st.metric("Total Tables", summary.get("total_tables", 0))

# Research text
st.subheader("Research Content")
st.write(summary.get("text", "No research content available"))

# Evaluation details
st.divider()
st.header("Statistical Evaluation")

evaluation = data.get("evaluation_summary", {})

st.write("Test Name:", evaluation.get("test_name"))
st.write("P-value:", evaluation.get("p_value"))
st.write("Confidence Interval:", evaluation.get("confidence_interval"))
st.write("Verdict:", evaluation.get("verdict"))

# Processing status
st.divider()
st.header("Processing Status")

st.success(data.get("processing_status", "Unknown"))
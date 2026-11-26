Module 5: Synthesis, Dashboard and Evaluation

Overview

Module 5 is responsible for synthesizing research paper information, displaying statistical evaluation results, and presenting the processed information through an interactive web dashboard.

This module is part of the AI-Powered Autonomous Discovery System, which uses LLMs, Multi-Agent Architecture, RAG, and Reinforcement Learning to support autonomous scientific research.

Main Components

1. Research Synthesis

- Reads extracted research paper information.
- Generates a structured research summary.
- Displays paper ID, extracted text, total images, and total tables.
- Saves the synthesis results in JSON format.

File: "synthesis.py"

Output: "synthesis_output.json"

2. Statistical Evaluation

- Reads statistical evaluation information.
- Displays test name, p-value, confidence interval, and verdict.
- Identifies missing evaluation fields.
- Saves evaluation results in JSON format.

File: "evaluation.py"

Output: "evaluation_output.json"

3. Research Synthesis Dashboard

- Provides a web-based interface using Streamlit.
- Displays research paper details.
- Shows extracted content and image/table counts.
- Presents statistical evaluation results.
- Displays processing status.

File: "dashboard.py"

Project Files

- "read_input.py" – Reads input JSON data.
- "sample_input.json" – Sample research paper input.
- "synthesis.py" – Generates research summaries.
- "synthesis_output.json" – Stores synthesis results.
- "evaluation.py" – Processes statistical evaluation information.
- "evaluation_output.json" – Stores evaluation results.
- "dashboard.py" – Displays results through a Streamlit dashboard.

Installation

Install the required library:

python -m pip install streamlit

Run the Dashboard

From the project root directory, run:

python -m streamlit run "module 5/dashboard.py"

The dashboard will open in your web browser.

Current Status

Research synthesis, statistical evaluation output generation, and the Streamlit dashboard have been implemented and tested using sample input data.

Note: Integration with the actual outputs from other modules will be updated when their final data formats are available.
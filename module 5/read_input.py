import json
from pathlib import Path

# Read JSON file from the same folder
file_path = Path(__file__).parent / "sample_input.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract paper details
paper_id = data["paper_id"]

# Extract content details
extracted_data = data["extracted_content"]

text_content = extracted_data.get("text", "")
images = extracted_data.get("images", [])
tables = extracted_data.get("tables", [])

# Extract statistical evaluation
stats = data.get("statistical_evaluation", {})

test_name = stats.get("test_name", "Not available")
p_value = stats.get("p_value")
confidence_interval = stats.get("confidence_interval")
verdict = stats.get("verdict", "Not available")

# Display summary
print("===== MODULE 5: DOCUMENT SUMMARY =====")

print("Paper ID:", paper_id)

print("\n===== EXTRACTION DETAILS =====")
print("Extracted Text:", text_content)
print("Total Images:", len(images))
print("Total Tables:", len(tables))

print("\n===== STATISTICAL EVALUATION =====")
print("Test Name:", test_name)
print("P-value:", p_value)
print("Confidence Interval:", confidence_interval)
print("Verdict:", verdict)
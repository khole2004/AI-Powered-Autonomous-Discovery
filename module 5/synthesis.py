import json
from pathlib import Path


def generate_synthesis(input_data):

    # Extract paper information
    paper_id = input_data.get("paper_id", "Unknown")

    # Extract research content
    content = input_data.get("extracted_content", {})

    text = content.get("text", "")
    images = content.get("images", [])
    tables = content.get("tables", [])

    # Extract statistical evaluation
    evaluation = input_data.get("statistical_evaluation", {})

    # Generate research summary
    synthesis = {
        "paper_id": paper_id,
        "research_summary": {
            "text": text,
            "total_images": len(images),
            "total_tables": len(tables)
        },
        "evaluation_summary": {
            "test_name": evaluation.get("test_name"),
            "p_value": evaluation.get("p_value"),
            "confidence_interval": evaluation.get(
                "confidence_interval"
            ),
            "verdict": evaluation.get("verdict", "pending")
        },
        "processing_status": "completed"
    }

    return synthesis


def main():

    folder = Path(__file__).parent
    input_file = folder / "sample_input.json"
    output_file = folder / "synthesis_output.json"

    # Read input JSON
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Generate synthesis
    result = generate_synthesis(data)

    # Save synthesis output
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("Research Synthesis Generated Successfully!")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
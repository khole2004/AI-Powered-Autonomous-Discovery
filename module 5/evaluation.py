import json
from pathlib import Path


def evaluate_research(synthesis_data):

    evaluation = synthesis_data.get(
        "evaluation_summary", {}
    )

    p_value = evaluation.get("p_value")
    confidence_interval = evaluation.get(
        "confidence_interval"
    )
    verdict = evaluation.get("verdict", "pending")

    missing_fields = []

    if p_value is None:
        missing_fields.append("p_value")

    if confidence_interval is None:
        missing_fields.append("confidence_interval")

    if verdict == "pending":
        evaluation_status = "Pending"
    elif missing_fields:
        evaluation_status = "Incomplete"
    else:
        evaluation_status = "Available"

    report = {
        "paper_id": synthesis_data.get("paper_id"),
        "evaluation_status": evaluation_status,
        "test_name": evaluation.get("test_name"),
        "p_value": p_value,
        "confidence_interval": confidence_interval,
        "verdict": verdict,
        "missing_fields": missing_fields
    }

    return report


def main():

    folder = Path(__file__).parent

    input_file = folder / "synthesis_output.json"
    output_file = folder / "evaluation_output.json"

    with open(input_file, "r", encoding="utf-8") as file:
        synthesis_data = json.load(file)

    report = evaluate_research(synthesis_data)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Evaluation Report Generated Successfully!")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
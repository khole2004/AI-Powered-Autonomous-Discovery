"""
Module 4 - Statistical Critic Agent.

The Critic Agent is the single entry point for reviewing an experiment's
results. It does not do any statistics itself - it COORDINATES the two
modules that do, and makes sure every outcome (including bad input and failed
experiments) comes back in one safe, predictable shape.

Workflow
--------

    experiment result (dict, or path to a JSON file)
          |
          v
    1. stats_checks.parse_execution_result()   the ADAPTER - the only step
          |                                    that knows the input schema
          |-- invalid input ---------------> outcome "invalid_input"
          v
    2. experiment reported as failed? -------> outcome "experiment_failed"
          |                                    (metrics are NOT reviewed)
          v
    3. stats_checks.run_checks()               deterministic statistics
          v                                    (SciPy / statsmodels)
    4. report.build_critic_report()            verdict + explanation
          v
       outcome "reviewed"

    Any unexpected exception in steps 3-4 -> outcome "internal_error"
    (reported with its details, never hidden, never turned into a verdict).

Design rules
------------
* No LLM is used anywhere in the agent. Every statistical value comes from
  `stats_checks` (SciPy / statsmodels). Nothing is estimated or invented.
* The agent imports only `stats_checks` and `report`. It does not import the
  PDF parser, figure extractor or VLM client - those modules stay independent.
* `verdict` is filled ONLY when a review actually ran (outcome "reviewed").
  For every other outcome it is None, so "not reviewed" can never be mistaken
  for "no issues".
* The input is never modified.

Input schema
------------
Still the TEMPORARY TEST SCHEMA `module4-temporary-test-schema-v0` from
stats_checks.py (including its optional "status" / "error" fields). Module 3's
ExecutionResult has not been received; no compatibility with it is claimed.
Mock inputs for testing live in tests/fixtures/mock_execution_results/.

Example
-------
    from critic.critic_agent import CriticAgent

    agent = CriticAgent()                       # default settings
    result = agent.review("results.json", job_id="job-42")
    print(result["outcome"], result["verdict"])
    print(result["message"])

Command line:

    python src/critic/critic_agent.py                         # built-in example
    python src/critic/critic_agent.py a.json b.json --out-dir reports
"""

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:                                    # normal use: `from critic import ...`
    from critic import report as rp
    from critic import stats_checks as sc
except ImportError:                     # run directly: python src/critic/critic_agent.py
    import report as rp
    import stats_checks as sc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_VERSION = "module4-critic-agent-v0"

# Outcomes: what happened to the input (not what the statistics concluded).
REVIEWED = "reviewed"                    # checks ran, report produced
EXPERIMENT_FAILED = "experiment_failed"  # input says the run failed
INVALID_INPUT = "invalid_input"          # could not be read / wrong schema
INTERNAL_ERROR = "internal_error"        # unexpected bug in Module 4

OUTCOMES = (REVIEWED, EXPERIMENT_FAILED, INVALID_INPUT, INTERNAL_ERROR)

RESULT_KEYS = ("agent_version", "outcome", "verdict", "message", "job_id",
               "job_id_source", "experiment_status", "is_temporary_schema",
               "source", "generated_at", "critic_report", "errors", "notes")


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class CriticAgent:
    """
    Coordinates statistical checks and Critic Report generation.

    Parameters (all optional; passed straight to stats_checks):
        alpha          significance level, 0 < alpha < 1        (default 0.05)
        target_effect  Cohen's d used for power analysis, > 0   (default 0.5)
        target_power   desired power, 0 < power < 1             (default 0.80)
        min_sample     small-sample warning threshold, >= 1     (default 30)

    Invalid settings raise ValueError immediately: they are a programming
    mistake, not a property of the experiment, so failing fast is safest.
    """

    def __init__(self, alpha=sc.DEFAULT_ALPHA,
                 target_effect=sc.DEFAULT_TARGET_EFFECT,
                 target_power=sc.DEFAULT_TARGET_POWER,
                 min_sample=sc.DEFAULT_MIN_SAMPLE):
        _require_number("alpha", alpha, low=0, high=1)
        _require_number("target_power", target_power, low=0, high=1)
        _require_number("target_effect", target_effect, low=0)
        if (isinstance(min_sample, bool) or not isinstance(min_sample, int)
                or min_sample < 1):
            raise ValueError(f"min_sample must be a whole number >= 1, "
                             f"got {min_sample!r}.")
        self.settings = {"alpha": alpha, "target_effect": target_effect,
                         "target_power": target_power, "min_sample": min_sample}

    # ----- public API -----------------------------------------------------

    def review(self, execution_result, job_id=None, generated_at=None):
        """
        Review one experiment result. NEVER raises for bad input.

        `execution_result` is a dict in the temporary schema, or a path
        (str / Path) to a JSON file containing one.

        Returns the AGENT RESULT dict (keys in RESULT_KEYS):

            outcome        "reviewed" | "experiment_failed" |
                           "invalid_input" | "internal_error"
            verdict        the Critic Report verdict - ONLY when reviewed,
                           otherwise None
            message        one plain-language summary of what happened
            critic_report  the full report.py report, or None
            errors         list of {"stage", "type", "message", ...}
            notes          caveats about how the result was produced
            ... plus job_id, job_id_source, experiment_status,
                is_temporary_schema, source, generated_at, agent_version
        """
        if isinstance(execution_result, (str, Path)):
            return self.review_file(execution_result, job_id, generated_at)
        return self._review_data(execution_result, job_id, generated_at,
                                 source=None)

    def review_file(self, path, job_id=None, generated_at=None):
        """Review a JSON file. A missing or unreadable file -> invalid_input."""
        generated_at = generated_at or _now()
        try:
            data = sc.load_execution_result(path)
        except (FileNotFoundError, sc.SchemaError) as error:
            return _result(INVALID_INPUT, generated_at, source=str(path),
                           job_id=job_id,
                           message=_invalid_message(error),
                           errors=[_error("input", error)])
        return self._review_data(data, job_id, generated_at, source=str(path))

    def review_many(self, items, generated_at=None):
        """Review several inputs (dicts or paths). One result per input."""
        return [self.review(item, generated_at=generated_at) for item in items]

    # ----- the workflow ---------------------------------------------------

    def _review_data(self, data, job_id, generated_at, source):
        generated_at = generated_at or _now()

        # Step 1 - adapter: the only place the input schema is understood.
        try:
            internal = sc.parse_execution_result(data)
        except sc.SchemaError as error:
            return _result(INVALID_INPUT, generated_at, source=source,
                           job_id=job_id, message=_invalid_message(error),
                           errors=[_error("input", error)])

        common = {"source": source, "job_id": job_id,
                  "experiment_id": internal["experiment_id"],
                  "experiment_status": internal["status"],
                  "is_temporary_schema": internal["is_temporary_schema"]}

        # Step 2 - a failed run's numbers are not reviewed.
        if internal["status"] == "failed":
            return _failed_experiment_result(internal, generated_at, common)

        # Steps 3-4 - statistics, then the report.
        stage = "checks"
        try:
            stats_report = sc.run_checks(internal, **self.settings)
            stage = "report"
            critic_report = rp.build_critic_report(
                stats_report, job_id=job_id, generated_at=generated_at)
        except Exception as error:      # a bug in Module 4, not in the input
            return _result(
                INTERNAL_ERROR, generated_at, **common,
                message=(f"An unexpected error occurred during the {stage} "
                         f"step, so no verdict is given. This indicates a "
                         f"problem in Module 4 itself, not necessarily in "
                         f"the experiment: {type(error).__name__}: {error}"),
                errors=[_error(stage, error, include_traceback=True)])

        notes = []
        if internal["status"] is None:
            notes.append("The experiment's status was not reported; the "
                         "results were reviewed as given.")

        return _result(
            REVIEWED, generated_at, **common,
            verdict=critic_report["verdict"],
            message=critic_report["explanation"],
            critic_report=critic_report,
            # Report-derived identifiers, so agent and report always agree.
            resolved_job_id=(critic_report["job_id"],
                             critic_report["job_id_source"]),
            notes=notes)


# ---------------------------------------------------------------------------
# Result building helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_number(name, value, low=None, high=None):
    """Validate a numeric setting (bool is rejected even though it's an int)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    if (low is not None and value <= low) or (high is not None and value >= high):
        bounds = f"between {low} and {high}" if high is not None else f"above {low}"
        raise ValueError(f"{name} must be {bounds} (exclusive), got {value}.")


def _error(stage, error, include_traceback=False):
    """A structured error record."""
    record = {"stage": stage, "type": type(error).__name__,
              "message": str(error)}
    if include_traceback:
        record["traceback"] = traceback.format_exc()
    return record


def _invalid_message(error):
    return (f"The input could not be read as a temporary-schema "
            f"ExecutionResult, so nothing was reviewed and no verdict is "
            f"given: {error}")


def _resolve_job_id(job_id, experiment_id):
    """Same rule as report.py: provided > experiment_id > None (never invented)."""
    if job_id:
        return job_id, "provided"
    if experiment_id:
        return experiment_id, "experiment_id (temporary schema)"
    return None, None


def _failed_experiment_result(internal, generated_at, common):
    ignored = {"metrics": len(internal["metrics"]),
               "comparisons": len(internal["comparisons"])}
    reason = (f" (reported error: {internal['error'].strip().rstrip('.')})"
              if internal["error"] else "")
    message = (f"The experiment is reported as failed{reason}. Its results "
               f"were not statistically reviewed, because numbers from a "
               f"failed run cannot be relied on; no verdict is given.")
    notes = []
    if ignored["metrics"] or ignored["comparisons"]:
        notes.append(f"Not reviewed: {ignored['metrics']} metric(s) and "
                     f"{ignored['comparisons']} comparison(s) present in the "
                     "failed run.")
    return _result(
        EXPERIMENT_FAILED, generated_at, **common, message=message,
        errors=[{"stage": "experiment", "type": "ExperimentFailed",
                 "message": internal["error"] or "No error description given."}],
        notes=notes)


def _result(outcome, generated_at, message, source=None, job_id=None,
            experiment_id=None, experiment_status=None,
            is_temporary_schema=None, verdict=None, critic_report=None,
            errors=None, notes=None, resolved_job_id=None):
    """Assemble the agent result. Every outcome has exactly RESULT_KEYS."""
    if resolved_job_id is None:
        resolved_job_id = _resolve_job_id(job_id, experiment_id)
    return {
        "agent_version": AGENT_VERSION,
        "outcome": outcome,
        "verdict": verdict if outcome == REVIEWED else None,
        "message": message,
        "job_id": resolved_job_id[0],
        "job_id_source": resolved_job_id[1],
        "experiment_status": experiment_status,
        "is_temporary_schema": is_temporary_schema,
        "source": source,
        "generated_at": generated_at,
        "critic_report": critic_report,
        "errors": errors or [],
        "notes": notes or [],
    }


# ---------------------------------------------------------------------------
# Convenience functions and output
# ---------------------------------------------------------------------------

def review_execution_result(execution_result, job_id=None, **settings):
    """One-call shortcut: CriticAgent(**settings).review(...)."""
    return CriticAgent(**settings).review(execution_result, job_id=job_id)


def result_to_json(result, indent=2):
    """The agent result as a JSON string."""
    return json.dumps(result, indent=indent, ensure_ascii=False)


def save_result(result, path):
    """Write the agent result to a JSON file (folders created). Returns the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result_to_json(result), encoding="utf-8")
    return path


def format_result_text(result):
    """Human-readable rendering of an agent result."""
    header = [f"Critic Agent - outcome: {result['outcome'].upper()}"
              + (f"   source: {result['source']}" if result["source"] else "")]
    if result["critic_report"] is not None:
        body = [rp.format_report_text(result["critic_report"])]
    else:
        body = ["=" * 76,
                f"NO REVIEW   job: {result['job_id'] or '(none)'}   "
                f"verdict: (none - not reviewed)",
                "=" * 76,
                result["message"]]
        for error in result["errors"]:
            body.append(f"  [{error['stage']}] {error['type']}: {error['message']}")
    extra = [f"  * {note}" for note in result["notes"]]
    return "\n".join(header + body + (["Agent notes:"] + extra if extra else []))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None):
    """
    Review one or more JSON files (or the built-in example).

    Exit code 0 when every input was processed (reviewed or recorded as a
    failed experiment) - even if issues were found, because finding issues is
    the agent working correctly. Exit code 1 when any input was invalid or an
    internal error occurred.
    """
    parser = argparse.ArgumentParser(
        description="Statistical Critic Agent (temporary input schema).")
    parser.add_argument("inputs", nargs="*",
                        help="temporary-schema JSON files (default: built-in example)")
    parser.add_argument("--job-id", default=None,
                        help="job identifier (only with a single input)")
    parser.add_argument("--out-dir", default=None,
                        help="save each result as <name>_critic.json here")
    parser.add_argument("--alpha", type=float, default=sc.DEFAULT_ALPHA)
    parser.add_argument("--target-effect", type=float, default=sc.DEFAULT_TARGET_EFFECT)
    parser.add_argument("--target-power", type=float, default=sc.DEFAULT_TARGET_POWER)
    parser.add_argument("--min-sample", type=int, default=sc.DEFAULT_MIN_SAMPLE)
    args = parser.parse_args(argv)

    if args.job_id and len(args.inputs) > 1:
        parser.error("--job-id can only be used with a single input file.")

    try:
        agent = CriticAgent(alpha=args.alpha, target_effect=args.target_effect,
                            target_power=args.target_power,
                            min_sample=args.min_sample)
    except ValueError as error:
        parser.error(str(error))

    if args.inputs:
        jobs = [(Path(p).stem, p) for p in args.inputs]
    else:
        print("No input files given - using the built-in TEMPORARY-schema "
              "example.\n")
        jobs = [("example", sc.EXAMPLE_EXECUTION_RESULT)]

    exit_code = 0
    for name, item in jobs:
        result = agent.review(item, job_id=args.job_id)
        print(format_result_text(result).encode("ascii", "replace").decode("ascii"))
        if args.out_dir:
            saved = save_result(result, Path(args.out_dir) / f"{name}_critic.json")
            print(f"\nSaved: {saved}")
        print()
        if result["outcome"] in (INVALID_INPUT, INTERNAL_ERROR):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

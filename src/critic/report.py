"""
Module 4 - Statistical Critic: Critic Report generation.

Turns the output of `stats_checks.run_checks()` (a flat list of check
results) into a structured CRITIC REPORT: one entry per metric/comparison,
each with its key numbers, a verdict, a plain-language explanation and the
individual check results it is based on, plus an overall verdict for the job.

Pipeline (the input schema is only ever seen by the adapter):

    raw input --[ stats_checks.parse_execution_result ]--> internal format
              --[ stats_checks.run_checks ]--> stats report
              --[ report.build_critic_report ]--> CRITIC REPORT

Because this module consumes the STATS REPORT, not the raw input, it does not
depend on any input schema. The input is still the TEMPORARY TEST SCHEMA
(see stats_checks.py) - Module 3's ExecutionResult has not been received, and
no compatibility with it is claimed.

----------------------------------------------------------------------------
How verdicts avoid misleading conclusions
----------------------------------------------------------------------------
The checks test INTERNAL STATISTICAL CONSISTENCY (does the p-value match the
statistic? does the interval contain its estimate? was the study powered?).
They cannot show that a finding is true. So:

  * No verdict ever says "valid", "correct" or "significant".
  * The best possible verdict is "no_issues_detected", and its explanation
    always says this is not a confirmation.
  * Trivial checks (values in range, sample-size threshold) do not count as
    evidence on their own: an entry where only those ran is "inconclusive".
  * Anything not checked is listed explicitly under "not_checked".
  * Fields that cannot be filled are None and listed in "unavailable_fields";
    nothing is estimated to fill a gap.

Entry verdicts:
    issues_found        at least one check failed (definite problem)
    concerns            no failures, but at least one warning
    no_issues_detected  at least one substantive check ran; none flagged
    inconclusive        no substantive check could run

Overall (job) verdict: the same four, plus
    incomplete          some entries had no issues, others were inconclusive

Run directly to see a report for the built-in temporary-schema example:

    python src/critic/report.py [execution_result.json] [--job-id ID]
                                [--json OUTPUT.json]
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

try:                                    # normal use: `from critic import report`
    from critic import stats_checks as sc
except ImportError:                     # run directly: python src/critic/report.py
    import stats_checks as sc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Version of THIS report format (Module 4's own output, also provisional).
REPORT_VERSION = "module4-critic-report-v0"

ISSUES_FOUND = "issues_found"
CONCERNS = "concerns"
NO_ISSUES_DETECTED = "no_issues_detected"
INCOMPLETE = "incomplete"           # overall verdict only
INCONCLUSIVE = "inconclusive"

ENTRY_VERDICTS = (ISSUES_FOUND, CONCERNS, NO_ISSUES_DETECTED, INCONCLUSIVE)

# Checks that actually test a result. value_ranges and sample_size can still
# FAIL/WARN (and those count), but a PASS from them alone proves little.
SUBSTANTIVE_CHECKS = {"p_value_consistency", "degrees_of_freedom",
                      "confidence_interval", "statistical_power"}

NO_ISSUES_CAVEAT = (
    "This does not confirm that the result is correct or that the experiment "
    "was sound; it only means the checks that could run found no "
    "inconsistencies.")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_stats_report(stats_report):
    """Make sure we were given a stats_checks report, not something else."""
    if not isinstance(stats_report, dict):
        raise ValueError("Expected a stats_checks report (a dict) from "
                         "stats_checks.run_checks().")
    required = ("checks", "records", "settings", "schema_version",
                "is_temporary_schema")
    missing = [key for key in required if key not in stats_report]
    if missing:
        raise ValueError(f"Not a stats_checks report: missing {missing}. "
                         "Build it with stats_checks.run_checks().")
    records = stats_report["records"]
    if (not isinstance(stats_report["checks"], list)
            or not isinstance(records, dict)
            or not isinstance(records.get("metrics"), list)
            or not isinstance(records.get("comparisons"), list)):
        raise ValueError("Malformed stats_checks report: 'checks' must be a "
                         "list and 'records' must hold 'metrics' and "
                         "'comparisons' lists.")


# ---------------------------------------------------------------------------
# Per-entry fields
# ---------------------------------------------------------------------------

def _find_check(checks, name):
    """The first check result with this check name, or None."""
    return next((c for c in checks if c["check"] == name), None)


def _p_value_info(record, checks):
    """Reported p-value plus the recomputation, or None if none was reported."""
    if record.get("p_value") is None:
        return None
    p_check = _find_check(checks, "p_value_consistency")
    recomputed = None
    if p_check is not None and p_check["status"] != sc.SKIPPED:
        recomputed = p_check["details"].get("recomputed_p")
    return {
        "reported": record["p_value"],
        "operator": record.get("p_operator") or "=",
        "recomputed": recomputed,
        "consistency": p_check["status"] if p_check else sc.SKIPPED,
    }


def _confidence_interval_info(record):
    """The interval as reported, or None if the record has no interval."""
    if record.get("ci_lower") is None and record.get("ci_upper") is None:
        return None
    return {"estimate": record.get("value"),
            "lower": record.get("ci_lower"),
            "upper": record.get("ci_upper"),
            "level": record.get("ci_level")}


def _effect_size_info(kind, record, checks):
    """
    Derived Cohen's d / d_z for t-tests, or None.

    Only derived when the record's values passed the range check - an effect
    size computed from an impossible sample size would be meaningless.
    """
    if kind != "comparison":
        return None
    ranges_ok = all(c["status"] == sc.PASS for c in checks
                    if c["check"] == "value_ranges")
    if not ranges_ok:
        return None
    return sc.cohens_d_from_t(record.get("test"), record.get("statistic"),
                              record.get("n_a"), record.get("n_b"))


def _entry_verdict(checks):
    """Verdict for one entry from its check results (see module docstring)."""
    statuses = [c["status"] for c in checks]
    if sc.FAIL in statuses:
        return ISSUES_FOUND
    if sc.WARNING in statuses:
        return CONCERNS
    if any(c["status"] == sc.PASS and c["check"] in SUBSTANTIVE_CHECKS
           for c in checks):
        return NO_ISSUES_DETECTED
    return INCONCLUSIVE


def _entry_explanation(verdict, problems, concerns, not_checked):
    """Plain-language explanation built only from what the checks said."""
    if verdict == ISSUES_FOUND:
        parts = [f"{len(problems)} problem(s) found: " + " ".join(problems)]
        if concerns:
            parts.append(f"Also {len(concerns)} concern(s): " + " ".join(concerns))
    elif verdict == CONCERNS:
        parts = [f"No definite errors, but {len(concerns)} concern(s) need "
                 "review: " + " ".join(concerns)]
    elif verdict == NO_ISSUES_DETECTED:
        parts = ["The checks that could run found no problems. " + NO_ISSUES_CAVEAT]
    else:
        parts = ["Not enough information to assess this result, so no "
                 "conclusion is drawn."]
    if not_checked:
        parts.append("Not checked: " + " ".join(not_checked))
    return " ".join(parts)


def _build_entry(kind, record, checks):
    """One report entry for a metric or a comparison."""
    name = record.get("name")
    entry = {
        "target": f"{kind}:{name}",
        "kind": kind,
        "name": name,
        "test_name": record.get("test") if kind == "comparison" else None,
        "p_value": _p_value_info(record, checks),
        "confidence_interval": _confidence_interval_info(record),
        "effect_size": _effect_size_info(kind, record, checks),
    }
    entry["unavailable_fields"] = [
        field for field in ("test_name", "p_value", "confidence_interval",
                            "effect_size")
        if entry[field] is None and not (field == "test_name" and kind == "metric")
    ]

    problems = [c["message"] for c in checks if c["status"] == sc.FAIL]
    concerns = [c["message"] for c in checks if c["status"] == sc.WARNING]
    not_checked = [c["message"] for c in checks if c["status"] == sc.SKIPPED]
    verdict = _entry_verdict(checks)

    entry.update({
        "verdict": verdict,
        "explanation": _entry_explanation(verdict, problems, concerns, not_checked),
        "problems": problems,
        "concerns": concerns,
        "not_checked": not_checked,
        "checks": checks,
    })
    return entry


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------

def _overall_verdict(entries):
    """Job-level verdict from the entry verdicts."""
    verdicts = [e["verdict"] for e in entries]
    if not verdicts:
        return INCONCLUSIVE
    if ISSUES_FOUND in verdicts:
        return ISSUES_FOUND
    if CONCERNS in verdicts:
        return CONCERNS
    if all(v == NO_ISSUES_DETECTED for v in verdicts):
        return NO_ISSUES_DETECTED
    if NO_ISSUES_DETECTED in verdicts:
        return INCOMPLETE
    return INCONCLUSIVE


def _overall_explanation(verdict, entries):
    """Job-level explanation; always says what could not be assessed."""
    count = {v: [e["name"] for e in entries if e["verdict"] == v]
             for v in ENTRY_VERDICTS}
    total = len(entries)

    if total == 0:
        return ("The input contained no metrics or comparisons, so nothing "
                "could be assessed and no conclusion is drawn.")

    if verdict == ISSUES_FOUND:
        text = (f"Problems were found in {len(count[ISSUES_FOUND])} of {total} "
                f"result(s): {', '.join(count[ISSUES_FOUND])}.")
        if count[CONCERNS]:
            text += (f" {len(count[CONCERNS])} other result(s) raised concerns: "
                     f"{', '.join(count[CONCERNS])}.")
    elif verdict == CONCERNS:
        text = (f"No definite errors, but {len(count[CONCERNS])} of {total} "
                f"result(s) raised concerns: {', '.join(count[CONCERNS])}.")
    elif verdict == NO_ISSUES_DETECTED:
        text = (f"No problems were detected in any of the {total} result(s). "
                + NO_ISSUES_CAVEAT)
    elif verdict == INCOMPLETE:
        text = (f"No problems were detected in "
                f"{len(count[NO_ISSUES_DETECTED])} of {total} result(s), but "
                f"{len(count[INCONCLUSIVE])} could not be assessed, so the job "
                "as a whole is not fully checked. " + NO_ISSUES_CAVEAT)
    else:
        text = ("None of the results contained enough information to be "
                "assessed, so no conclusion is drawn.")

    if verdict in (ISSUES_FOUND, CONCERNS) and count[INCONCLUSIVE]:
        text += (f" {len(count[INCONCLUSIVE])} result(s) could not be assessed: "
                 f"{', '.join(count[INCONCLUSIVE])}.")
    elif verdict == INCOMPLETE:
        text += f" Not assessed: {', '.join(count[INCONCLUSIVE])}."
    return text


def _notes(stats_report, entries):
    """Standing caveats that apply to the whole report."""
    notes = []
    if stats_report["is_temporary_schema"]:
        notes.append(
            f"Input used the TEMPORARY test schema "
            f"'{stats_report['schema_version']}'. It is not Module 3's "
            "ExecutionResult, and no compatibility with Module 3 is implied.")
    if any(e["effect_size"] for e in entries):
        notes.append("Effect sizes are DERIVED from t statistics and sample "
                     "sizes; they were not reported in the input.")
    settings = stats_report["settings"]
    if any(c["check"] == "statistical_power"
           for e in entries for c in e["checks"]):
        notes.append(
            f"Statistical power is computed for a target effect of "
            f"d = {settings.get('target_effect')} chosen in advance, not from "
            "the observed effect (post-hoc 'observed power' is not used).")
    notes.append("Verdicts describe internal statistical consistency only; "
                 "they do not establish whether the findings are true.")
    return notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_critic_report(stats_report, job_id=None, generated_at=None):
    """
    Build a Critic Report from a stats_checks report.

    Parameters
    ----------
    stats_report : dict    output of stats_checks.run_checks() /
                           run_stats_checks()
    job_id : str, optional The job identifier. If not given, the input's
                           experiment_id is used and `job_id_source` says so.
                           If neither exists it is None - never invented.
    generated_at : str, optional  ISO timestamp; defaults to now (UTC).
                           Pass a fixed value for reproducible output.

    Returns
    -------
    dict:
        {
          "report_version", "job_id", "job_id_source", "generated_at",
          "schema_version", "is_temporary_schema", "settings",
          "verdict", "explanation",
          "summary": {"entries": n, <verdict>: n, ..., "checks": {...}},
          "entries": [
            {"target", "kind", "name", "test_name", "p_value",
             "confidence_interval", "effect_size", "unavailable_fields",
             "verdict", "explanation", "problems", "concerns",
             "not_checked", "checks": [individual check results]}
          ],
          "notes": [standing caveats]
        }

    The input is never modified.
    """
    _validate_stats_report(stats_report)
    stats_report = copy.deepcopy(stats_report)

    # Group check results by the record they belong to.
    by_target = {}
    for check in stats_report["checks"]:
        by_target.setdefault(check["target"], []).append(check)

    entries = []
    for kind, key in (("metric", "metrics"), ("comparison", "comparisons")):
        for record in stats_report["records"][key]:
            target = f"{kind}:{record.get('name')}"
            entries.append(_build_entry(kind, record, by_target.pop(target, [])))

    # Defensive: checks whose target matches no record are kept, not dropped.
    for target, checks in by_target.items():
        kind, _, name = target.partition(":")
        entries.append(_build_entry(kind or "unknown", {"name": name}, checks))

    if job_id:
        job_id_source = "provided"
    elif stats_report.get("experiment_id"):
        job_id = stats_report["experiment_id"]
        job_id_source = "experiment_id (temporary schema)"
    else:
        job_id, job_id_source = None, None

    verdict = _overall_verdict(entries)
    summary = {"entries": len(entries)}
    summary.update({v: sum(e["verdict"] == v for e in entries)
                    for v in ENTRY_VERDICTS})
    summary["checks"] = stats_report.get("summary")

    return {
        "report_version": REPORT_VERSION,
        "job_id": job_id,
        "job_id_source": job_id_source,
        "generated_at": generated_at or datetime.now(timezone.utc)
                                                .isoformat(timespec="seconds"),
        "schema_version": stats_report["schema_version"],
        "is_temporary_schema": stats_report["is_temporary_schema"],
        "settings": stats_report["settings"],
        "verdict": verdict,
        "explanation": _overall_explanation(verdict, entries),
        "summary": summary,
        "entries": entries,
        "notes": _notes(stats_report, entries),
    }


def generate_critic_report(execution_result, job_id=None, generated_at=None,
                           **settings):
    """
    Convenience: raw TEMPORARY-schema input -> Critic Report in one call.

    Goes through stats_checks' adapter, so it accepts only the temporary
    schema and raises stats_checks.SchemaError for anything else. `settings`
    (alpha, target_effect, target_power, min_sample) go to the checks.
    """
    stats_report = sc.run_stats_checks(execution_result, **settings)
    return build_critic_report(stats_report, job_id=job_id,
                               generated_at=generated_at)


def report_to_json(report, indent=2):
    """The report as a JSON string."""
    return json.dumps(report, indent=indent, ensure_ascii=False)


def save_report(report, path):
    """Write the report to a JSON file (folders created). Returns the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report), encoding="utf-8")
    return path


def format_report_text(report):
    """A human-readable, plain-text rendering of the report."""
    labels = {ISSUES_FOUND: "ISSUES FOUND", CONCERNS: "CONCERNS",
              NO_ISSUES_DETECTED: "NO ISSUES DETECTED", INCOMPLETE: "INCOMPLETE",
              INCONCLUSIVE: "INCONCLUSIVE"}
    lines = ["=" * 76,
             f"CRITIC REPORT   job: {report['job_id'] or '(none)'}"
             + (f"   [{report['job_id_source']}]" if report["job_id_source"] else ""),
             f"Verdict: {labels[report['verdict']]}",
             "=" * 76,
             report["explanation"], ""]

    for entry in report["entries"]:
        lines.append(f"--- {entry['kind']}: {entry['name']}  "
                     f"-> {labels[entry['verdict']]}")
        if entry["test_name"]:
            lines.append(f"    test               : {entry['test_name']}")
        if entry["p_value"]:
            p = entry["p_value"]
            recomputed = (f", recomputed {p['recomputed']}"
                          if p["recomputed"] is not None else "")
            lines.append(f"    p-value            : {p['operator']} "
                         f"{p['reported']}{recomputed}")
        if entry["confidence_interval"]:
            ci = entry["confidence_interval"]
            level = f"{ci['level']:.0%} " if ci["level"] else ""
            lines.append(f"    confidence interval: {ci['estimate']} "
                         f"({level}CI {ci['lower']} to {ci['upper']})")
        if entry["effect_size"]:
            es = entry["effect_size"]
            approx = " (approximate)" if es["approximate"] else ""
            lines.append(f"    effect size        : {es['measure']} = "
                         f"{es['value']}{approx}, derived")
        if entry["unavailable_fields"]:
            lines.append(f"    not available      : "
                         f"{', '.join(entry['unavailable_fields'])}")
        lines.append(f"    {entry['explanation']}")
        lines.append("")

    lines.append("Notes:")
    lines.extend(f"  * {note}" for note in report["notes"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Critic Report from a TEMPORARY-schema "
                    "ExecutionResult JSON file.")
    parser.add_argument("input", nargs="?",
                        help="temporary-schema JSON (default: built-in example)")
    parser.add_argument("--job-id", default=None, help="job identifier")
    parser.add_argument("--json", default=None, help="also save the report here")
    args = parser.parse_args()

    try:
        if args.input:
            source = sc.load_execution_result(args.input)
        else:
            print("No file given - using the built-in TEMPORARY-schema example.\n")
            source = sc.EXAMPLE_EXECUTION_RESULT
        critic_report = generate_critic_report(source, job_id=args.job_id)
    except (FileNotFoundError, sc.SchemaError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

    print(format_report_text(critic_report).encode("ascii", "replace")
          .decode("ascii"))
    if args.json:
        print(f"\nJSON report saved to: {save_report(critic_report, args.json)}")

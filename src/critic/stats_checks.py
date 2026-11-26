"""
Module 4 - Statistical Critic: statistical checks on experiment results.

What this module checks (first version, scope agreed with the team lead):

  1. Reported-value consistency
       * p-values recomputed from the test statistic and degrees of freedom
         (t, z, F, chi-square) and compared with the reported p-value;
       * degrees of freedom consistent with the sample sizes;
       * confidence intervals that actually contain their point estimate;
       * impossible values (p outside 0-1, sample size below 1, ...).
  2. Sample size and power
       * small-sample warnings (a configurable heuristic threshold);
       * power to detect a TARGET effect size, the smallest effect the study
         could reliably detect, and the sample size that would be needed.

Deliberately NOT done: "observed" (post-hoc) power computed from the effect
the study itself found. That number is a one-to-one transformation of the
p-value, so it adds no information and is widely discouraged. Power here is
always computed for a target effect size chosen in advance.

============================================================================
  IMPORTANT - TEMPORARY INPUT SCHEMA - NOT THE MODULE 3 CONTRACT
============================================================================
These checks are intended, eventually, to run on Module 3's `ExecutionResult`
metrics. That schema was NOT available (searched: local code, every GitHub
branch, issues, pull requests and wiki), so this module makes NO claim of
compatibility with Module 3. It reads only a TEMPORARY TEST SCHEMA, defined
here by Module 4 for testing, and identified by:

    "schema_version": "module4-temporary-test-schema-v0"

Any other schema_version is rejected with an error - nothing is guessed.

Three layers keep the input format away from the statistics:

    raw input --[ parse_execution_result ]--> INTERNAL FORMAT --[ run_checks ]--> report
                  (adapter: the ONLY code that                (statistics: never sees
                   knows the input schema)                     the input schema)

  * Statistical functions (check_p_value, check_power, ...) take plain numbers.
  * `run_checks()` takes the INTERNAL FORMAT (documented at
    `parse_execution_result`), not the input schema.
  * `run_stats_checks()` is just `run_checks(parse_execution_result(raw))`.

When Module 3's real schema arrives, write/replace the adapter so it returns
the same INTERNAL FORMAT; nothing else needs to change.

Temporary test schema (v0):

    {
      "schema_version": "module4-temporary-test-schema-v0",   (required)
      "experiment_id": "exp-001",                              (optional string)
      "status": "completed",            (optional: "completed" or "failed";
                                         omitted means "not reported")
      "error": "CUDA out of memory",    (optional string, for failed runs)
      "metrics": [                                             (optional list)
        {"name": "accuracy",           (required)
         "value": 0.91,                (optional number)
         "n": 200,                     (optional integer >= 1)
         "ci_lower": 0.87,             (optional, with ci_upper)
         "ci_upper": 0.95,
         "ci_level": 0.95}             (optional fraction, e.g. 0.95)
      ],
      "comparisons": [                                         (optional list)
        {"name": "model_a_vs_baseline",  (required)
         "test": "two_sample_t",         (required, see SUPPORTED_TESTS)
         "statistic": 2.31,              (optional number)
         "df": 98,                       (t / chi2 / F numerator df)
         "df2": null,                    (F denominator df)
         "p_value": 0.023,               (optional, 0-1)
         "p_operator": "=",              ("=", "<", "<=", ">"; default "=")
         "tails": 2,                     (1 or 2; default 2)
         "n_a": 50, "n_b": 50}           (group sizes; n_a only for one-sample)
      ]
    }

Unknown keys are rejected too, so a typo such as "pvalue" fails loudly
instead of silently skipping a check.
============================================================================

Run directly for a demonstration on a built-in example (or pass a JSON file):

    python src/critic/stats_checks.py [execution_result.json]
"""

import json
import math
import sys
from decimal import Decimal
from pathlib import Path

from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEMP_SCHEMA_VERSION = "module4-temporary-test-schema-v0"

DEFAULT_ALPHA = 0.05            # significance level
DEFAULT_TARGET_EFFECT = 0.5     # Cohen's d; 0.2 small, 0.5 medium, 0.8 large
DEFAULT_TARGET_POWER = 0.80     # conventional minimum power
DEFAULT_MIN_SAMPLE = 30         # per-group "small sample" heuristic

# Check outcomes, from best to worst.
PASS = "pass"          # check ran and found no problem
WARNING = "warning"    # questionable, worth a human look
FAIL = "fail"          # definite inconsistency or impossible value
SKIPPED = "skipped"    # not enough information to run the check

# Report-level only: nothing could be checked at all.
INCONCLUSIVE = "inconclusive"

# Test names accepted in "test", mapped to the distribution used to
# recompute p, and to the study design used for power analysis.
SUPPORTED_TESTS = {
    #  name             family   power design
    "t":              ("t",     None),
    "one_sample_t":   ("t",     "one_sample"),
    "paired_t":       ("t",     "one_sample"),   # paired = one-sample on diffs
    "two_sample_t":   ("t",     "two_sample"),   # Student's (equal variances)
    "welch_t":        ("t",     "two_sample"),
    "z":              ("z",     None),
    "f":              ("f",     None),
    "anova":          ("f",     None),
    "chi2":           ("chi2",  None),
    "chi_square":     ("chi2",  None),
}

P_OPERATORS = ("=", "<", "<=", ">")

# Optional run status in the temporary schema. Omitted = "not reported".
EXPERIMENT_STATUSES = ("completed", "failed")

METRIC_FIELDS = {"name", "value", "n", "ci_lower", "ci_upper", "ci_level"}
COMPARISON_FIELDS = {"name", "test", "statistic", "df", "df2", "p_value",
                     "p_operator", "tails", "n_a", "n_b"}


class SchemaError(ValueError):
    """The input is not a valid temporary-schema ExecutionResult."""


def _check(check, target, status, message, **details):
    """Build one check result in the standard shape."""
    return {"check": check, "target": target, "status": status,
            "message": message, "details": details}


def _is_number(value):
    """True for int/float, but NOT bool (True is technically an int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ===========================================================================
# 1. Reported-value consistency
# ===========================================================================

def compute_p_value(test, statistic, df=None, df2=None, tails=2):
    """
    Recompute a p-value from a test statistic.

    t and z tests honour `tails` (a one-tailed p assumes the statistic lies
    in the hypothesised direction). F and chi-square tests are one-tailed by
    nature, so `tails` is ignored for them.

    Raises ValueError for an unsupported test or missing degrees of freedom.
    """
    if test not in SUPPORTED_TESTS:
        raise ValueError(f"unsupported test '{test}'")
    family = SUPPORTED_TESTS[test][0]

    if family == "t":
        if df is None:
            raise ValueError("a t-test needs 'df'")
        p = stats.t.sf(abs(statistic), df) * tails
    elif family == "z":
        p = stats.norm.sf(abs(statistic)) * tails
    elif family == "f":
        if df is None or df2 is None:
            raise ValueError("an F-test needs 'df' and 'df2'")
        p = stats.f.sf(statistic, df, df2)
    else:  # chi2
        if df is None:
            raise ValueError("a chi-square test needs 'df'")
        p = stats.chi2.sf(statistic, df)

    return float(min(p, 1.0))


def reported_decimals(value):
    """
    How many decimal places a reported number was written with.

    0.023 -> 3, 0.02 -> 2, 1e-05 -> 5. A p-value reported as 0.023 could be
    anything from 0.0225 to 0.0235, so the tolerance must match the rounding.
    """
    exponent = Decimal(repr(float(value))).as_tuple().exponent
    return max(0, -exponent)


def check_p_value(test, statistic, reported_p, df=None, df2=None, tails=2,
                  p_operator="=", alpha=DEFAULT_ALPHA, target=""):
    """
    Compare a reported p-value with one recomputed from the test statistic.

    Outcome:
        PASS    - consistent, allowing for the rounding of the reported value
        WARNING - inconsistent, but the significant / not-significant
                  conclusion at `alpha` is the same either way
        FAIL    - inconsistent AND the conclusion changes
        SKIPPED - unsupported test, missing df, or invalid inputs
    """
    name = "p_value_consistency"

    if statistic is None or reported_p is None:
        return _check(name, target, SKIPPED,
                      "Needs both a test statistic and a reported p-value.")
    if not 0 <= reported_p <= 1:
        return _check(name, target, SKIPPED,
                      "Reported p-value is outside 0-1 (see value_ranges).")
    try:
        recomputed = compute_p_value(test, statistic, df, df2, tails)
    except ValueError as error:
        return _check(name, target, SKIPPED, f"Cannot recompute p: {error}.")

    details = {"test": test, "statistic": statistic, "df": df, "df2": df2,
               "tails": tails, "reported_p": reported_p, "p_operator": p_operator,
               "recomputed_p": round(recomputed, 6), "alpha": alpha}

    # Is the recomputed value compatible with what was written?
    if p_operator == "=":
        tolerance = 0.5 * 10 ** -reported_decimals(reported_p)
        consistent = abs(recomputed - reported_p) <= tolerance + 1e-12
        details["rounding_tolerance"] = tolerance
    elif p_operator in ("<", "<="):
        consistent = recomputed < reported_p + 1e-12
    else:  # ">"
        consistent = recomputed > reported_p - 1e-12

    if consistent:
        return _check(name, target, PASS,
                      f"Reported p {p_operator} {reported_p} matches the "
                      f"recomputed p = {recomputed:.4g}.", **details)

    # Inconsistent: does it change the conclusion?
    if p_operator == "=":
        reported_significant = reported_p < alpha
    elif p_operator in ("<", "<="):
        reported_significant = reported_p <= alpha
    else:
        reported_significant = False
    recomputed_significant = recomputed < alpha
    details["conclusion_changes"] = reported_significant != recomputed_significant

    if details["conclusion_changes"]:
        return _check(name, target, FAIL,
                      f"Reported p {p_operator} {reported_p} but the statistic "
                      f"gives p = {recomputed:.4g}; this changes the "
                      f"conclusion at alpha = {alpha}.", **details)
    return _check(name, target, WARNING,
                  f"Reported p {p_operator} {reported_p} does not match the "
                  f"recomputed p = {recomputed:.4g} (same conclusion at "
                  f"alpha = {alpha}).", **details)


def check_degrees_of_freedom(test, df, n_a=None, n_b=None, target=""):
    """
    Check that t-test degrees of freedom fit the sample sizes.

        two_sample_t        df == n_a + n_b - 2
        paired / one_sample df == n_a - 1
        welch_t             min(n_a, n_b) - 1 <= df <= n_a + n_b - 2

    A mismatch is a WARNING, not a FAIL: excluded data points or a
    different test variant can legitimately explain it.
    """
    name = "degrees_of_freedom"
    if df is None or n_a is None:
        return _check(name, target, SKIPPED, "Needs 'df' and sample size(s).")

    if test == "two_sample_t":
        if n_b is None:
            return _check(name, target, SKIPPED, "two_sample_t needs n_a and n_b.")
        low = high = n_a + n_b - 2
    elif test in ("paired_t", "one_sample_t"):
        low = high = n_a - 1
    elif test == "welch_t":
        if n_b is None:
            return _check(name, target, SKIPPED, "welch_t needs n_a and n_b.")
        low, high = min(n_a, n_b) - 1, n_a + n_b - 2
    else:
        return _check(name, target, SKIPPED,
                      f"No df rule for test '{test}'.")

    details = {"test": test, "df": df, "n_a": n_a, "n_b": n_b,
               "expected_min": low, "expected_max": high}
    if low - 1e-9 <= df <= high + 1e-9:
        return _check(name, target, PASS,
                      f"df = {df} is consistent with the sample sizes.", **details)
    expected = f"{low}" if low == high else f"{low}-{high}"
    return _check(name, target, WARNING,
                  f"df = {df} but the sample sizes imply {expected}; data may "
                  f"have been excluded, or a different test used.", **details)


def check_confidence_interval(value, ci_lower, ci_upper, ci_level=None, target=""):
    """
    Check a confidence interval is well-formed and contains its estimate.
    """
    name = "confidence_interval"
    if ci_lower is None and ci_upper is None:
        return _check(name, target, SKIPPED, "No confidence interval given.")
    if ci_lower is None or ci_upper is None:
        return _check(name, target, FAIL,
                      "Only one confidence-interval bound was given.",
                      ci_lower=ci_lower, ci_upper=ci_upper)

    details = {"value": value, "ci_lower": ci_lower, "ci_upper": ci_upper,
               "ci_level": ci_level}

    if ci_lower > ci_upper:
        return _check(name, target, FAIL,
                      f"Lower bound {ci_lower} is above upper bound {ci_upper}.",
                      **details)
    if ci_level is not None and not 0 < ci_level < 1:
        return _check(name, target, FAIL,
                      f"ci_level = {ci_level} is not a fraction between 0 and 1 "
                      f"(write 0.95, not 95).", **details)
    if value is None:
        return _check(name, target, PASS,
                      "Interval is well-formed (no point estimate to compare).",
                      **details)
    if not ci_lower <= value <= ci_upper:
        return _check(name, target, FAIL,
                      f"Estimate {value} lies outside its interval "
                      f"[{ci_lower}, {ci_upper}].", **details)
    return _check(name, target, PASS,
                  f"Estimate {value} lies inside [{ci_lower}, {ci_upper}].",
                  **details)


def check_value_ranges(record, target=""):
    """
    Flag values that are impossible regardless of context.

    Returns a list: one FAIL per impossible value, or a single PASS.
    """
    name = "value_ranges"
    problems = []

    p = record.get("p_value")
    if p is not None and not 0 <= p <= 1:
        problems.append(f"p_value = {p} is outside 0-1")

    for field in ("n", "n_a", "n_b"):
        n = record.get(field)
        if n is not None and (n < 1 or n != int(n)):
            problems.append(f"{field} = {n} is not a whole number >= 1")

    for field in ("df", "df2"):
        df = record.get(field)
        if df is not None and df <= 0:
            problems.append(f"{field} = {df} must be greater than 0")

    tails = record.get("tails")
    if tails is not None and tails not in (1, 2):
        problems.append(f"tails = {tails} must be 1 or 2")

    test = record.get("test")
    statistic = record.get("statistic")
    if test in SUPPORTED_TESTS and statistic is not None:
        if SUPPORTED_TESTS[test][0] in ("f", "chi2") and statistic < 0:
            problems.append(f"{test} statistic = {statistic} cannot be negative")

    if not problems:
        return [_check(name, target, PASS, "All values are within valid ranges.")]
    return [_check(name, target, FAIL, problem) for problem in problems]


def cohens_d_from_t(test, statistic, n_a=None, n_b=None):
    """
    Derive a standardised effect size from a t statistic and sample sizes.

    Standard conversions (Lakens, 2013; Cohen, 1988):
        two_sample_t     d   = t * sqrt(1/n_a + 1/n_b)       (exact)
        welch_t          d   = t * sqrt(1/n_a + 1/n_b)       (approximation)
        paired_t,
        one_sample_t     d_z = t / sqrt(n_a)                  (exact)

    Returns a dict, or None when no justified conversion exists (non-t tests,
    the generic "t" whose design is unknown, or missing/invalid inputs):

        {"value": 0.548, "measure": "cohens_d" | "cohens_dz",
         "approximate": bool,
         "source": "derived from the t statistic and sample sizes"}

    The sign follows the t statistic. This is DERIVED, not a reported value.
    """
    if statistic is None or n_a is None or n_a < 2:
        return None

    if test in ("two_sample_t", "welch_t"):
        if n_b is None or n_b < 2:
            return None
        value = statistic * math.sqrt(1 / n_a + 1 / n_b)
        measure, approximate = "cohens_d", test == "welch_t"
    elif test in ("paired_t", "one_sample_t"):
        value = statistic / math.sqrt(n_a)
        measure, approximate = "cohens_dz", False
    else:
        return None

    return {"value": round(value, 4), "measure": measure,
            "approximate": approximate,
            "source": "derived from the t statistic and sample sizes"}


# ===========================================================================
# 2. Sample size and power
# ===========================================================================

def check_sample_size(n, min_sample=DEFAULT_MIN_SAMPLE, target="", label="n"):
    """
    WARN when a sample is below `min_sample`.

    This is a rule-of-thumb heuristic, not a statistical law: what counts as
    "too small" depends on the effect and the test. The power check below is
    the more principled companion.
    """
    name = "sample_size"
    if n is None:
        return _check(name, target, SKIPPED, f"No {label} given.")
    details = {label: n, "min_sample": min_sample}
    if n < min_sample:
        return _check(name, target, WARNING,
                      f"{label} = {n} is below the small-sample threshold of "
                      f"{min_sample}; results may be unstable.", **details)
    return _check(name, target, PASS,
                  f"{label} = {n} meets the threshold of {min_sample}.", **details)


def check_power(design, n_a, n_b=None, alpha=DEFAULT_ALPHA,
                target_effect=DEFAULT_TARGET_EFFECT,
                target_power=DEFAULT_TARGET_POWER, tails=2, target=""):
    """
    Power analysis for a t-test design, for a TARGET effect size.

    design: "two_sample" (independent groups, n_a and n_b) or
            "one_sample" (one-sample or paired, n_a only).

    Reports, all for the given alpha and number of tails:
        power_at_target_effect  - chance of detecting d = target_effect
        minimum_detectable_effect - smallest d detectable at target_power
        required_n              - per group (two-sample) or total pairs
                                  (one-sample) needed for target_power

    WARNING if power_at_target_effect < target_power.
    """
    name = "statistical_power"
    alternative = "two-sided" if tails == 2 else "larger"

    if design not in ("two_sample", "one_sample"):
        return _check(name, target, SKIPPED,
                      "Power analysis is only implemented for t-test designs.")
    if n_a is None or (design == "two_sample" and n_b is None):
        return _check(name, target, SKIPPED, "Sample size(s) missing.")
    if n_a < 2 or (design == "two_sample" and n_b < 2):
        return _check(name, target, SKIPPED,
                      "Each group needs at least 2 observations.")

    if design == "two_sample":
        analysis = TTestIndPower()
        ratio = n_b / n_a
        common = {"alpha": alpha, "ratio": ratio, "alternative": alternative}
        power = analysis.power(effect_size=target_effect, nobs1=n_a, **common)
        mde = analysis.solve_power(nobs1=n_a, power=target_power, **common)
        required = analysis.solve_power(effect_size=target_effect,
                                        power=target_power, **common)
        required_label = "required_n_per_group"
    else:
        analysis = TTestPower()
        common = {"alpha": alpha, "alternative": alternative}
        power = analysis.power(effect_size=target_effect, nobs=n_a, **common)
        mde = analysis.solve_power(nobs=n_a, power=target_power, **common)
        required = analysis.solve_power(effect_size=target_effect,
                                        power=target_power, **common)
        required_label = "required_n"

    details = {
        "design": design, "n_a": n_a, "n_b": n_b, "alpha": alpha, "tails": tails,
        "target_effect": target_effect, "target_power": target_power,
        "power_at_target_effect": round(float(power), 4),
        "minimum_detectable_effect": round(float(mde), 4),
        required_label: int(math.ceil(float(required))),
    }

    if power < target_power:
        return _check(name, target, WARNING,
                      f"Power to detect d = {target_effect} is only "
                      f"{power:.0%} (target {target_power:.0%}); the smallest "
                      f"reliably detectable effect is d = {mde:.2f}. About "
                      f"{details[required_label]} would be needed "
                      f"({required_label.replace('_', ' ')}).", **details)
    return _check(name, target, PASS,
                  f"Power to detect d = {target_effect} is {power:.0%} "
                  f"(target {target_power:.0%}).", **details)


# ===========================================================================
# Adapter for the TEMPORARY schema  (the only schema-aware code)
# ===========================================================================

def load_execution_result(path):
    """Read an ExecutionResult JSON file (temporary schema)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ExecutionResult file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        # e.g. a PDF or other binary file passed by mistake
        raise SchemaError(
            f"{path} is not a UTF-8 text/JSON file: {error}") from error
    except OSError as error:
        raise SchemaError(f"{path} could not be read: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise SchemaError(f"{path} is not valid JSON: {error}") from error


def _validate_record(record, allowed, kind, index):
    """Shared structural checks for one metric/comparison record."""
    where = f"{kind}[{index}]"
    if not isinstance(record, dict):
        raise SchemaError(f"{where} must be an object, got {type(record).__name__}.")

    unknown = set(record) - allowed
    if unknown:
        raise SchemaError(
            f"{where} has unknown field(s) {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}.")

    if not isinstance(record.get("name"), str) or not record["name"].strip():
        raise SchemaError(f"{where} needs a non-empty string 'name'.")

    for field, value in record.items():
        if field in ("name", "test", "p_operator") or value is None:
            continue
        if not _is_number(value):
            raise SchemaError(
                f"{where} ('{record['name']}'): '{field}' must be a number, "
                f"got {value!r}.")
        # Python's json module accepts NaN / Infinity, which are not valid
        # reported values. Reject them instead of letting them reach SciPy.
        if not math.isfinite(value):
            raise SchemaError(
                f"{where} ('{record['name']}'): '{field}' must be a finite "
                f"number, got {value!r}. Use null for a missing value.")


def _reject_duplicate_names(records, kind):
    """Each name must be unique within its list - the report keys by name."""
    seen = {}
    for i, record in enumerate(records):
        name = record["name"].strip()
        if name in seen:
            raise SchemaError(
                f"{kind}[{i}] repeats the name '{name}' already used by "
                f"{kind}[{seen[name]}]. Names must be unique within '{kind}'.")
        seen[name] = i


def parse_execution_result(data):
    """
    ADAPTER: convert a TEMPORARY-schema ExecutionResult to the INTERNAL FORMAT.

    This is the ONLY function that knows the input format. It is NOT a
    Module 3 adapter - Module 3's schema has not been received. When it is,
    replace or add an adapter that returns the same INTERNAL FORMAT below.

    INTERNAL FORMAT (what `run_checks` consumes; every key always present,
    missing values are None):

        {
          "schema_version": str,         where the data came from
          "is_temporary_schema": bool,   True for this adapter
          "experiment_id": str | None,
          "status": "completed" | "failed" | None,   None = not reported
          "error": str | None,           failure description, if any
          "metrics": [
            {"name", "value", "n", "ci_lower", "ci_upper", "ci_level"}
          ],
          "comparisons": [
            {"name", "test" (a SUPPORTED_TESTS key), "statistic", "df", "df2",
             "p_value", "p_operator" (one of P_OPERATORS, default "="),
             "tails" (default 2), "n_a", "n_b"}
          ]
        }

    Raises SchemaError for anything that is not the temporary schema.
    """
    if not isinstance(data, dict):
        raise SchemaError("ExecutionResult must be a JSON object.")

    version = data.get("schema_version")
    if version != TEMP_SCHEMA_VERSION:
        raise SchemaError(
            f"Unsupported schema_version {version!r}. Only the temporary test "
            f"schema '{TEMP_SCHEMA_VERSION}' is supported until Module 3's "
            f"real ExecutionResult schema is available. Nothing was guessed.")

    unknown_top = set(data) - {"schema_version", "experiment_id", "status",
                               "error", "metrics", "comparisons"}
    if unknown_top:
        raise SchemaError(f"Unknown top-level field(s): {sorted(unknown_top)}.")

    experiment_id = data.get("experiment_id")
    if experiment_id is not None and not isinstance(experiment_id, str):
        raise SchemaError(
            f"'experiment_id' must be a string, got {experiment_id!r}.")

    status = data.get("status")
    if status is not None and status not in EXPERIMENT_STATUSES:
        raise SchemaError(
            f"'status' must be one of {EXPERIMENT_STATUSES} (or omitted), "
            f"got {status!r}.")

    error = data.get("error")
    if error is not None and not isinstance(error, str):
        raise SchemaError(f"'error' must be a string, got {error!r}.")

    metrics = data.get("metrics", [])
    comparisons = data.get("comparisons", [])
    for key, value in (("metrics", metrics), ("comparisons", comparisons)):
        if not isinstance(value, list):
            raise SchemaError(f"'{key}' must be a list.")

    normalised_metrics = []
    for i, record in enumerate(metrics):
        _validate_record(record, METRIC_FIELDS, "metrics", i)
        normalised_metrics.append({f: record.get(f) for f in METRIC_FIELDS})
    _reject_duplicate_names(metrics, "metrics")

    normalised_comparisons = []
    for i, record in enumerate(comparisons):
        _validate_record(record, COMPARISON_FIELDS, "comparisons", i)
        test = record.get("test")
        if test not in SUPPORTED_TESTS:
            raise SchemaError(
                f"comparisons[{i}] ('{record['name']}'): unsupported test "
                f"{test!r}. Supported: {sorted(SUPPORTED_TESTS)}.")
        operator = record.get("p_operator", "=")
        if operator not in P_OPERATORS:
            raise SchemaError(
                f"comparisons[{i}] ('{record['name']}'): p_operator must be "
                f"one of {P_OPERATORS}, got {operator!r}.")
        normalised = {f: record.get(f) for f in COMPARISON_FIELDS}
        normalised["p_operator"] = operator
        normalised["tails"] = record.get("tails", 2)
        normalised_comparisons.append(normalised)
    _reject_duplicate_names(comparisons, "comparisons")

    return {"schema_version": version,
            "is_temporary_schema": True,
            "experiment_id": experiment_id,
            "status": status,
            "error": error,
            "metrics": normalised_metrics,
            "comparisons": normalised_comparisons}


# ===========================================================================
# Run everything
# ===========================================================================

def run_stats_checks(execution_result, alpha=DEFAULT_ALPHA,
                     target_effect=DEFAULT_TARGET_EFFECT,
                     target_power=DEFAULT_TARGET_POWER,
                     min_sample=DEFAULT_MIN_SAMPLE):
    """
    Convenience entry point: adapter + checks in one call (see `run_checks`
    for the report format).

    Exactly `run_checks(parse_execution_result(execution_result), ...)`.
    Accepts only the TEMPORARY test schema (see module docstring).
    """
    return run_checks(parse_execution_result(execution_result), alpha=alpha,
                      target_effect=target_effect, target_power=target_power,
                      min_sample=min_sample)


def run_checks(data, alpha=DEFAULT_ALPHA,
               target_effect=DEFAULT_TARGET_EFFECT,
               target_power=DEFAULT_TARGET_POWER,
               min_sample=DEFAULT_MIN_SAMPLE):
    """
    Run every applicable check on data in the INTERNAL FORMAT.

    `data` must come from an adapter such as `parse_execution_result`; this
    function never looks at the raw input schema.

    Returns a report dictionary:
        {"schema_version", "is_temporary_schema", "experiment_id",
         "settings": {...}, "checks": [ ... ],
         "records": {"metrics": [...], "comparisons": [...]},
         "summary": {"pass": n, "warning": n, "fail": n, "skipped": n},
         "overall": "pass" | "warning" | "fail" | "inconclusive"}

    `records` is a copy of the internal-format input the checks ran on, so a
    consumer (e.g. report.py) can show the original values next to the checks
    without ever seeing the raw input schema.

    `overall` is the worst status found. If nothing could actually be checked
    (no records, or every check skipped) it is "inconclusive" - never "pass",
    because an empty or unusable result must not look like a clean bill of
    health.
    """
    checks = []

    for metric in data["metrics"]:
        target = f"metric:{metric['name']}"
        checks.extend(check_value_ranges(metric, target))
        if metric["ci_lower"] is not None or metric["ci_upper"] is not None:
            checks.append(check_confidence_interval(
                metric["value"], metric["ci_lower"], metric["ci_upper"],
                metric["ci_level"], target))
        if metric["n"] is not None:
            checks.append(check_sample_size(metric["n"], min_sample, target))

    for comp in data["comparisons"]:
        target = f"comparison:{comp['name']}"
        ranges = check_value_ranges(comp, target)
        checks.extend(ranges)
        values_ok = all(r["status"] == PASS for r in ranges)

        checks.append(check_p_value(
            comp["test"], comp["statistic"], comp["p_value"], comp["df"],
            comp["df2"], comp["tails"] if values_ok else 2,
            comp["p_operator"], alpha, target))

        family, design = SUPPORTED_TESTS[comp["test"]]
        if family == "t" and comp["df"] is not None:
            checks.append(check_degrees_of_freedom(
                comp["test"], comp["df"], comp["n_a"], comp["n_b"], target))

        for label in ("n_a", "n_b"):
            if comp[label] is not None:
                checks.append(check_sample_size(comp[label], min_sample,
                                                target, label))

        if design is not None and values_ok:
            checks.append(check_power(
                design, comp["n_a"], comp["n_b"], alpha, target_effect,
                target_power, comp["tails"], target))

    summary = {status: 0 for status in (PASS, WARNING, FAIL, SKIPPED)}
    for check in checks:
        summary[check["status"]] += 1

    if summary[FAIL]:
        overall = FAIL
    elif summary[WARNING]:
        overall = WARNING
    elif summary[PASS]:
        overall = PASS
    else:
        overall = INCONCLUSIVE   # nothing was actually checked

    return {
        "schema_version": data["schema_version"],
        "is_temporary_schema": data["is_temporary_schema"],
        "experiment_id": data["experiment_id"],
        "settings": {"alpha": alpha, "target_effect": target_effect,
                     "target_power": target_power, "min_sample": min_sample},
        "checks": checks,
        "records": {
            "metrics": [dict(m) for m in data["metrics"]],
            "comparisons": [dict(c) for c in data["comparisons"]],
        },
        "summary": summary,
        "overall": overall,
    }


# ===========================================================================
# Demo
# ===========================================================================

# Hand-made example in the TEMPORARY schema, with deliberate problems so the
# demo shows every outcome. It is NOT data from Module 3.
EXAMPLE_EXECUTION_RESULT = {
    "schema_version": TEMP_SCHEMA_VERSION,
    "experiment_id": "demo-temporary-schema",
    "metrics": [
        {"name": "accuracy", "value": 0.91, "n": 200,
         "ci_lower": 0.87, "ci_upper": 0.95, "ci_level": 0.95},
        {"name": "f1_score", "value": 0.97, "n": 12,
         "ci_lower": 0.80, "ci_upper": 0.94, "ci_level": 0.95},
    ],
    "comparisons": [
        # Consistent and well powered.
        {"name": "model_a_vs_baseline", "test": "two_sample_t",
         "statistic": 3.10, "df": 126, "p_value": 0.002, "n_a": 64, "n_b": 64},
        # Wrong p-value that flips the conclusion; also under-powered.
        {"name": "model_b_vs_baseline", "test": "two_sample_t",
         "statistic": 1.20, "df": 38, "p_value": 0.03, "n_a": 20, "n_b": 20},
        # "p < .001" style report, consistent.
        {"name": "ablation_chi2", "test": "chi2", "statistic": 15.2, "df": 1,
         "p_value": 0.001, "p_operator": "<"},
    ],
}


def _print_report(report):
    """Human-readable report for the console."""
    print("=" * 76)
    print(f"Statistical checks - experiment {report['experiment_id']}")
    print(f"Input schema: {report['schema_version']}  (TEMPORARY TEST SCHEMA,")
    print("not Module 3's real ExecutionResult format)")
    print("=" * 76)
    marks = {PASS: "PASS", WARNING: "WARN", FAIL: "FAIL", SKIPPED: "SKIP"}
    for check in report["checks"]:
        print(f"[{marks[check['status']]}] {check['target']:<34} "
              f"{check['check']}")
        print(f"       {check['message']}")
    s = report["summary"]
    print("-" * 76)
    print(f"Summary: {s[PASS]} pass, {s[WARNING]} warning, {s[FAIL]} fail, "
          f"{s[SKIPPED]} skipped  ->  overall: {report['overall'].upper()}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            source = load_execution_result(sys.argv[1])
        except (FileNotFoundError, SchemaError) as error:
            print(f"ERROR: {error}")
            raise SystemExit(1)
    else:
        print("No file given - using the built-in TEMPORARY-schema example.\n")
        source = EXAMPLE_EXECUTION_RESULT

    try:
        _print_report(run_stats_checks(source))
    except SchemaError as error:
        print(f"SCHEMA ERROR: {error}")
        raise SystemExit(1)

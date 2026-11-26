"""
Tests for src/critic/stats_checks.py.

Expected statistical values come from PUBLISHED REFERENCE VALUES, not from
SciPy itself (re-using SciPy to produce the "expected" answer would only test
SciPy against itself):

  * critical values from standard statistical tables, e.g.
      t(df=10) = 2.228, z = 1.960, chi2(df=1) = 3.841, F(1, 10) = 4.965
    all give p = 0.05;
  * Cohen (1988), Statistical Power Analysis for the Behavioral Sciences:
      two independent groups, d = 0.5, alpha = .05 two-sided
      -> 64 per group for 80% power; d = 0.8 -> 26; d = 0.2 -> 393.

The adapter tests use the TEMPORARY test schema. Nothing here assumes or
tests compatibility with Module 3's ExecutionResult, which has not been
received.

Run from the project root:

    python -m unittest discover -s tests -v
"""

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

import pdf_fixtures as fx  # also puts src/ on sys.path

from critic import stats_checks as sc


# ---------------------------------------------------------------------------
# p-value recomputation
# ---------------------------------------------------------------------------

class TestComputePValue(unittest.TestCase):
    """Published critical values -> the p-value they correspond to."""

    def test_t_distribution(self):
        self.assertAlmostEqual(sc.compute_p_value("t", 2.228, df=10), 0.05, places=3)
        self.assertAlmostEqual(sc.compute_p_value("t", 2.086, df=20), 0.05, places=3)
        self.assertAlmostEqual(sc.compute_p_value("t", 3.169, df=10), 0.01, places=3)

    def test_t_sign_does_not_matter_for_two_tailed(self):
        self.assertAlmostEqual(sc.compute_p_value("two_sample_t", -2.228, df=10),
                               sc.compute_p_value("two_sample_t", 2.228, df=10))

    def test_one_tailed_halves_the_p_value(self):
        self.assertAlmostEqual(sc.compute_p_value("t", 2.228, df=10, tails=1),
                               0.025, places=3)

    def test_normal_distribution(self):
        self.assertAlmostEqual(sc.compute_p_value("z", 1.960), 0.05, places=3)
        self.assertAlmostEqual(sc.compute_p_value("z", 2.576), 0.01, places=3)

    def test_chi_square(self):
        self.assertAlmostEqual(sc.compute_p_value("chi2", 3.841, df=1), 0.05, places=3)
        self.assertAlmostEqual(sc.compute_p_value("chi_square", 5.991, df=2), 0.05, places=3)

    def test_f_distribution(self):
        self.assertAlmostEqual(sc.compute_p_value("f", 4.965, df=1, df2=10), 0.05, places=3)
        self.assertAlmostEqual(sc.compute_p_value("anova", 3.493, df=2, df2=20), 0.05, places=3)

    def test_missing_df_or_unknown_test_raises(self):
        with self.assertRaises(ValueError):
            sc.compute_p_value("t", 2.0)
        with self.assertRaises(ValueError):
            sc.compute_p_value("f", 2.0, df=1)
        with self.assertRaises(ValueError):
            sc.compute_p_value("mann_whitney", 2.0)


class TestReportedDecimals(unittest.TestCase):

    def test_examples(self):
        self.assertEqual(sc.reported_decimals(0.023), 3)
        self.assertEqual(sc.reported_decimals(0.02), 2)
        self.assertEqual(sc.reported_decimals(0.5), 1)
        self.assertEqual(sc.reported_decimals(1e-05), 5)


class TestCheckPValue(unittest.TestCase):

    def test_consistent_value_passes(self):
        result = sc.check_p_value("t", 2.228, 0.05, df=10)
        self.assertEqual(result["status"], sc.PASS)
        self.assertAlmostEqual(result["details"]["recomputed_p"], 0.05, places=3)

    def test_rounding_is_respected(self):
        # z = 1.96 gives p = 0.04999...; "0.05" (2 dp) is a correct rounding.
        self.assertEqual(sc.check_p_value("z", 1.96, 0.05)["status"], sc.PASS)

    def test_small_mismatch_same_conclusion_is_warning(self):
        # Reported 0.049 (3 dp, so 0.0485-0.0495) vs actual 0.04999:
        # wrong, but both are significant at .05.
        result = sc.check_p_value("z", 1.96, 0.049)
        self.assertEqual(result["status"], sc.WARNING)
        self.assertFalse(result["details"]["conclusion_changes"])

    def test_mismatch_that_creates_significance_fails(self):
        # z = 1.0 really gives p = 0.317; reporting 0.03 claims significance.
        result = sc.check_p_value("z", 1.0, 0.03)
        self.assertEqual(result["status"], sc.FAIL)
        self.assertTrue(result["details"]["conclusion_changes"])

    def test_mismatch_that_hides_significance_fails(self):
        # z = 2.576 really gives p = 0.01; reporting 0.20 hides it.
        self.assertEqual(sc.check_p_value("z", 2.576, 0.20)["status"], sc.FAIL)

    def test_less_than_operator(self):
        self.assertEqual(sc.check_p_value("z", 3.5, 0.001, p_operator="<")["status"],
                         sc.PASS)
        # Claims p < .001 but p is .046: wrong, still significant -> warning.
        self.assertEqual(sc.check_p_value("z", 2.0, 0.001, p_operator="<")["status"],
                         sc.WARNING)
        # Claims p < .05 but p is .317 -> conclusion changes.
        self.assertEqual(sc.check_p_value("z", 1.0, 0.05, p_operator="<")["status"],
                         sc.FAIL)

    def test_greater_than_operator(self):
        self.assertEqual(sc.check_p_value("z", 1.0, 0.05, p_operator=">")["status"],
                         sc.PASS)
        self.assertEqual(sc.check_p_value("z", 2.576, 0.05, p_operator=">")["status"],
                         sc.FAIL)

    def test_one_tailed(self):
        self.assertEqual(sc.check_p_value("t", 2.228, 0.025, df=10, tails=1)["status"],
                         sc.PASS)

    def test_custom_alpha_changes_the_verdict(self):
        # p reported 0.03, actual 0.0455 (z = 2.0): same side of 0.05 ...
        self.assertEqual(sc.check_p_value("z", 2.0, 0.03, alpha=0.05)["status"],
                         sc.WARNING)
        # ... but opposite sides of 0.04.
        self.assertEqual(sc.check_p_value("z", 2.0, 0.03, alpha=0.04)["status"],
                         sc.FAIL)

    def test_skipped_when_it_cannot_run(self):
        cases = [
            ("missing df", dict(test="t", statistic=2.0, reported_p=0.05)),
            ("missing statistic", dict(test="z", statistic=None, reported_p=0.05)),
            ("missing p", dict(test="z", statistic=2.0, reported_p=None)),
            ("p out of range", dict(test="z", statistic=2.0, reported_p=1.5)),
            ("unsupported test", dict(test="mann_whitney", statistic=2.0,
                                      reported_p=0.05)),
        ]
        for label, kwargs in cases:
            with self.subTest(case=label):
                self.assertEqual(sc.check_p_value(**kwargs)["status"], sc.SKIPPED)


# ---------------------------------------------------------------------------
# Degrees of freedom, confidence intervals, value ranges
# ---------------------------------------------------------------------------

class TestDegreesOfFreedom(unittest.TestCase):

    def test_two_sample(self):
        self.assertEqual(sc.check_degrees_of_freedom("two_sample_t", 98, 50, 50)["status"],
                         sc.PASS)
        self.assertEqual(sc.check_degrees_of_freedom("two_sample_t", 90, 50, 50)["status"],
                         sc.WARNING)

    def test_paired_and_one_sample(self):
        for test in ("paired_t", "one_sample_t"):
            with self.subTest(test=test):
                self.assertEqual(sc.check_degrees_of_freedom(test, 29, 30)["status"], sc.PASS)
                self.assertEqual(sc.check_degrees_of_freedom(test, 30, 30)["status"], sc.WARNING)

    def test_welch_allows_a_range(self):
        # n = 20 and 40 -> Welch df must lie between 19 and 58.
        self.assertEqual(sc.check_degrees_of_freedom("welch_t", 45.3, 20, 40)["status"], sc.PASS)
        self.assertEqual(sc.check_degrees_of_freedom("welch_t", 10, 20, 40)["status"], sc.WARNING)

    def test_skipped_without_enough_information(self):
        self.assertEqual(sc.check_degrees_of_freedom("two_sample_t", 98, 50)["status"], sc.SKIPPED)
        self.assertEqual(sc.check_degrees_of_freedom("z", 1, 50, 50)["status"], sc.SKIPPED)
        self.assertEqual(sc.check_degrees_of_freedom("t", None, 50, 50)["status"], sc.SKIPPED)


class TestConfidenceInterval(unittest.TestCase):

    def test_estimate_inside(self):
        self.assertEqual(sc.check_confidence_interval(0.91, 0.87, 0.95, 0.95)["status"], sc.PASS)

    def test_estimate_on_a_bound_is_inside(self):
        self.assertEqual(sc.check_confidence_interval(0.95, 0.87, 0.95)["status"], sc.PASS)

    def test_estimate_outside(self):
        self.assertEqual(sc.check_confidence_interval(0.97, 0.80, 0.94)["status"], sc.FAIL)

    def test_reversed_bounds(self):
        self.assertEqual(sc.check_confidence_interval(0.9, 0.95, 0.87)["status"], sc.FAIL)

    def test_level_given_as_percentage(self):
        result = sc.check_confidence_interval(0.9, 0.87, 0.95, ci_level=95)
        self.assertEqual(result["status"], sc.FAIL)
        self.assertIn("0.95", result["message"])

    def test_only_one_bound(self):
        self.assertEqual(sc.check_confidence_interval(0.9, 0.87, None)["status"], sc.FAIL)

    def test_no_interval_or_no_estimate(self):
        self.assertEqual(sc.check_confidence_interval(0.9, None, None)["status"], sc.SKIPPED)
        self.assertEqual(sc.check_confidence_interval(None, 0.87, 0.95)["status"], sc.PASS)


class TestValueRanges(unittest.TestCase):

    def statuses(self, record):
        return [r["status"] for r in sc.check_value_ranges(record)]

    def test_valid_record_gives_one_pass(self):
        self.assertEqual(self.statuses({"p_value": 0.03, "n_a": 20, "df": 38,
                                        "test": "t", "statistic": -2.1}),
                         [sc.PASS])

    def test_each_impossible_value_fails(self):
        cases = [{"p_value": 1.5}, {"p_value": -0.1}, {"n": 0}, {"n_a": 2.5},
                 {"df": 0}, {"df2": -3}, {"tails": 3},
                 {"test": "chi2", "statistic": -1.0},
                 {"test": "f", "statistic": -0.5}]
        for record in cases:
            with self.subTest(record=record):
                self.assertEqual(self.statuses(record), [sc.FAIL])

    def test_several_problems_are_all_reported(self):
        self.assertEqual(self.statuses({"p_value": 2, "n_a": 0, "df": -1}),
                         [sc.FAIL, sc.FAIL, sc.FAIL])


class TestCohensDFromT(unittest.TestCase):
    """Anchors chosen so the expected value is exact arithmetic."""

    def test_two_sample(self):
        # d = t * sqrt(1/50 + 1/50) = 2 * 0.2 = 0.4
        result = sc.cohens_d_from_t("two_sample_t", 2.0, 50, 50)
        self.assertAlmostEqual(result["value"], 0.4)
        self.assertEqual(result["measure"], "cohens_d")
        self.assertFalse(result["approximate"])
        self.assertIn("derived", result["source"])

    def test_paired_and_one_sample_give_dz(self):
        # d_z = t / sqrt(n) = 3 / 3 = 1.0
        for test in ("paired_t", "one_sample_t"):
            with self.subTest(test=test):
                result = sc.cohens_d_from_t(test, 3.0, 9)
                self.assertAlmostEqual(result["value"], 1.0)
                self.assertEqual(result["measure"], "cohens_dz")

    def test_welch_is_marked_approximate(self):
        self.assertTrue(sc.cohens_d_from_t("welch_t", 2.0, 50, 50)["approximate"])

    def test_sign_follows_the_statistic(self):
        self.assertAlmostEqual(sc.cohens_d_from_t("two_sample_t", -2.0, 50, 50)["value"],
                               -0.4)

    def test_no_justified_conversion_returns_none(self):
        cases = [("z", 2.0, 50, 50), ("chi2", 4.0, 50, None), ("f", 4.0, 50, 50),
                 ("t", 2.0, 50, 50),                 # design unknown
                 ("two_sample_t", 2.0, 50, None),    # n_b missing
                 ("two_sample_t", 2.0, 1, 50),       # n below 2
                 ("paired_t", None, 9, None)]        # no statistic
        for args in cases:
            with self.subTest(args=args):
                self.assertIsNone(sc.cohens_d_from_t(*args))


# ---------------------------------------------------------------------------
# Sample size and power
# ---------------------------------------------------------------------------

class TestSampleSize(unittest.TestCase):

    def test_threshold(self):
        self.assertEqual(sc.check_sample_size(12)["status"], sc.WARNING)
        self.assertEqual(sc.check_sample_size(30)["status"], sc.PASS)
        self.assertEqual(sc.check_sample_size(None)["status"], sc.SKIPPED)

    def test_custom_threshold(self):
        self.assertEqual(sc.check_sample_size(30, min_sample=50)["status"], sc.WARNING)


class TestPower(unittest.TestCase):
    """Reference values from Cohen (1988), alpha = .05 two-sided."""

    def test_cohen_64_per_group_gives_80_percent(self):
        result = sc.check_power("two_sample", 64, 64, target_effect=0.5)
        self.assertEqual(result["status"], sc.PASS)
        self.assertAlmostEqual(result["details"]["power_at_target_effect"], 0.80, places=2)
        self.assertEqual(result["details"]["required_n_per_group"], 64)
        self.assertAlmostEqual(result["details"]["minimum_detectable_effect"], 0.5, places=2)

    def test_cohen_required_n_for_large_and_small_effects(self):
        large = sc.check_power("two_sample", 10, 10, target_effect=0.8)
        small = sc.check_power("two_sample", 10, 10, target_effect=0.2)
        self.assertEqual(large["details"]["required_n_per_group"], 26)
        self.assertIn(small["details"]["required_n_per_group"], (393, 394))

    def test_cohen_power_table_n50_d05(self):
        # Cohen's table: n = 50 per group, d = 0.5 -> power .70
        result = sc.check_power("two_sample", 50, 50, target_effect=0.5)
        self.assertEqual(result["status"], sc.WARNING)
        self.assertAlmostEqual(result["details"]["power_at_target_effect"], 0.70, delta=0.015)

    def test_cohen_power_table_n20_d05(self):
        # Cohen's table: n = 20 per group, d = 0.5 -> power .33
        result = sc.check_power("two_sample", 20, 20, target_effect=0.5)
        self.assertEqual(result["status"], sc.WARNING)
        self.assertAlmostEqual(result["details"]["power_at_target_effect"], 0.33, delta=0.015)

    def test_one_sample_design(self):
        # One-sample / paired, d = 0.5: about 34 subjects for 80% power.
        result = sc.check_power("one_sample", 34, target_effect=0.5)
        self.assertEqual(result["status"], sc.PASS)
        self.assertEqual(result["details"]["required_n"], 34)

    def test_power_behaves_sensibly(self):
        def power(**kwargs):
            return sc.check_power(**kwargs)["details"]["power_at_target_effect"]
        equal_small = power(design="two_sample", n_a=30, n_b=30)
        unequal = power(design="two_sample", n_a=30, n_b=60)
        equal_large = power(design="two_sample", n_a=60, n_b=60)
        self.assertLess(equal_small, unequal)
        self.assertLess(unequal, equal_large)
        self.assertGreater(power(design="two_sample", n_a=30, n_b=30, tails=1),
                           equal_small)
        self.assertGreater(power(design="two_sample", n_a=30, n_b=30, target_effect=0.8),
                           equal_small)

    def test_skipped_when_it_cannot_run(self):
        cases = [("unknown design", dict(design=None, n_a=30, n_b=30)),
                 ("missing n_b", dict(design="two_sample", n_a=30)),
                 ("missing n_a", dict(design="one_sample", n_a=None)),
                 ("n below 2", dict(design="two_sample", n_a=1, n_b=30))]
        for label, kwargs in cases:
            with self.subTest(case=label):
                self.assertEqual(sc.check_power(**kwargs)["status"], sc.SKIPPED)


# ---------------------------------------------------------------------------
# Adapter for the TEMPORARY schema
# ---------------------------------------------------------------------------

def valid_input():
    """
    A small valid TEMPORARY-schema document that should pass EVERY check
    (fresh copy each call). 64 per group is Cohen's n for 80% power at
    d = 0.5; with 50 per group power is only ~70% and the power check
    (correctly) warns. t = 2.5 on df = 126 gives p = 0.0137 -> "0.014".
    """
    return {
        "schema_version": sc.TEMP_SCHEMA_VERSION,
        "experiment_id": "exp-test",
        "metrics": [{"name": "accuracy", "value": 0.9, "n": 100,
                     "ci_lower": 0.85, "ci_upper": 0.94, "ci_level": 0.95}],
        "comparisons": [{"name": "a_vs_b", "test": "two_sample_t",
                         "statistic": 2.5, "df": 126, "p_value": 0.014,
                         "n_a": 64, "n_b": 64}],
    }


class TestAdapter(unittest.TestCase):

    def test_output_is_the_internal_format(self):
        data = sc.parse_execution_result(valid_input())
        self.assertEqual(set(data), {"schema_version", "is_temporary_schema",
                                     "experiment_id", "status", "error",
                                     "metrics", "comparisons"})
        self.assertIs(data["is_temporary_schema"], True)
        self.assertEqual(set(data["metrics"][0]), sc.METRIC_FIELDS)
        self.assertEqual(set(data["comparisons"][0]), sc.COMPARISON_FIELDS)

    def test_defaults_are_filled_in(self):
        comp = sc.parse_execution_result(valid_input())["comparisons"][0]
        self.assertEqual(comp["p_operator"], "=")
        self.assertEqual(comp["tails"], 2)
        self.assertIsNone(comp["df2"])

    def test_metrics_and_comparisons_are_optional(self):
        data = sc.parse_execution_result({"schema_version": sc.TEMP_SCHEMA_VERSION})
        self.assertEqual((data["metrics"], data["comparisons"]), ([], []))
        self.assertIsNone(data["experiment_id"])

    def test_status_and_error_are_optional_and_not_assumed(self):
        data = sc.parse_execution_result(valid_input())
        # Omitted status means "not reported" - never silently "completed".
        self.assertIsNone(data["status"])
        self.assertIsNone(data["error"])

    def test_status_and_error_are_passed_through(self):
        doc = dict(valid_input(), status="failed", error="CUDA out of memory")
        data = sc.parse_execution_result(doc)
        self.assertEqual((data["status"], data["error"]),
                         ("failed", "CUDA out of memory"))
        completed = sc.parse_execution_result(dict(valid_input(), status="completed"))
        self.assertEqual(completed["status"], "completed")

    def test_invalid_status_error_or_experiment_id(self):
        for field, value in (("status", "crashed"), ("status", True),
                             ("error", 42), ("experiment_id", 123)):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(sc.SchemaError, field):
                    sc.parse_execution_result(dict(valid_input(), **{field: value}))

    def test_input_is_not_modified(self):
        original = valid_input()
        snapshot = copy.deepcopy(original)
        sc.parse_execution_result(original)
        self.assertEqual(original, snapshot)

    def test_other_schema_versions_are_rejected_not_guessed(self):
        # Including anything that might look like a Module 3 result: no
        # compatibility with Module 3 is assumed until its schema is received.
        for version in (None, "module3-v1", "1.0", "module4-temporary-test-schema-v1"):
            with self.subTest(version=version):
                doc = valid_input()
                doc["schema_version"] = version
                with self.assertRaisesRegex(sc.SchemaError, "Nothing was guessed"):
                    sc.parse_execution_result(doc)

    def test_structural_errors(self):
        def broken(change):
            doc = valid_input()
            change(doc)
            return doc

        cases = {
            "not an object": [1, 2, 3],
            "unknown top-level key": broken(lambda d: d.update(results=[])),
            "metrics not a list": broken(lambda d: d.update(metrics={"x": 1})),
            "record not an object": broken(lambda d: d["metrics"].append("acc")),
            "typo in field name": broken(lambda d: d["comparisons"][0].update(pvalue=0.01)),
            "missing name": broken(lambda d: d["metrics"][0].pop("name")),
            "number as text": broken(lambda d: d["comparisons"][0].update(p_value="0.014")),
            "boolean as number": broken(lambda d: d["metrics"][0].update(n=True)),
            "unsupported test": broken(lambda d: d["comparisons"][0].update(test="u_test")),
            "bad p_operator": broken(lambda d: d["comparisons"][0].update(p_operator="~")),
        }
        for label, doc in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(sc.SchemaError):
                    sc.parse_execution_result(doc)

    def test_non_finite_numbers_are_rejected(self):
        # json.loads accepts NaN / Infinity; they must never reach the checks.
        for value in (float("nan"), float("inf"), float("-inf")):
            for kind, field in (("metrics", "n"), ("metrics", "value"),
                                ("comparisons", "statistic"),
                                ("comparisons", "p_value")):
                with self.subTest(value=value, field=f"{kind}.{field}"):
                    doc = valid_input()
                    doc[kind][0][field] = value
                    with self.assertRaisesRegex(sc.SchemaError, "finite"):
                        sc.parse_execution_result(doc)

    def test_non_finite_numbers_from_json_file_are_rejected(self):
        # The literal tokens NaN / Infinity in a JSON file (Python parses them).
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                doc = valid_input()
                doc["metrics"][0]["n"] = "__PLACEHOLDER__"
                text = json.dumps(doc).replace('"__PLACEHOLDER__"', token)
                with self.assertRaisesRegex(sc.SchemaError, "finite"):
                    sc.parse_execution_result(json.loads(text))

    def test_duplicate_names_are_rejected(self):
        for kind in ("metrics", "comparisons"):
            with self.subTest(kind=kind):
                doc = valid_input()
                doc[kind].append(copy.deepcopy(doc[kind][0]))
                with self.assertRaisesRegex(sc.SchemaError, "unique"):
                    sc.parse_execution_result(doc)

    def test_same_name_in_metrics_and_comparisons_is_allowed(self):
        # Report targets are "metric:<name>" / "comparison:<name>", so these
        # do not collide.
        doc = valid_input()
        doc["metrics"][0]["name"] = doc["comparisons"][0]["name"]
        sc.parse_execution_result(doc)  # must not raise

    def test_typo_error_names_the_field(self):
        doc = valid_input()
        doc["comparisons"][0]["pvalue"] = 0.01
        with self.assertRaisesRegex(sc.SchemaError, "pvalue"):
            sc.parse_execution_result(doc)

    def test_schema_error_is_a_value_error(self):
        self.assertTrue(issubclass(sc.SchemaError, ValueError))


class TestLoadExecutionResult(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_json_file(self):
        path = self.dir / "result.json"
        path.write_text(json.dumps(valid_input()), encoding="utf-8")
        self.assertEqual(sc.load_execution_result(path), valid_input())

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            sc.load_execution_result(self.dir / "missing.json")

    def test_invalid_json(self):
        path = self.dir / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(sc.SchemaError):
            sc.load_execution_result(path)

    def test_binary_file_is_a_schema_error_not_a_crash(self):
        path = self.dir / "paper.pdf"
        path.write_bytes(b"%PDF-1.7\n\xff\xfe\x80\x81 binary \xc3\x28")
        with self.assertRaisesRegex(sc.SchemaError, "UTF-8"):
            sc.load_execution_result(path)


# ---------------------------------------------------------------------------
# Runner, and the adapter / statistics boundary
# ---------------------------------------------------------------------------

class TestRunChecks(unittest.TestCase):

    def test_demo_example_outcomes(self):
        report = sc.run_stats_checks(sc.EXAMPLE_EXECUTION_RESULT)
        self.assertEqual(report["summary"],
                         {"pass": 14, "warning": 4, "fail": 2, "skipped": 0})
        self.assertEqual(report["overall"], sc.FAIL)
        self.assertTrue(report["is_temporary_schema"])

        by_key = {(c["target"], c["check"]): c for c in report["checks"]
                  if c["check"] != "sample_size"}
        self.assertEqual(by_key[("metric:f1_score", "confidence_interval")]["status"],
                         sc.FAIL)
        p_check = by_key[("comparison:model_b_vs_baseline", "p_value_consistency")]
        self.assertEqual(p_check["status"], sc.FAIL)
        self.assertTrue(p_check["details"]["conclusion_changes"])
        self.assertEqual(by_key[("comparison:model_b_vs_baseline",
                                 "statistical_power")]["status"], sc.WARNING)

    def test_overall_pass_and_warning(self):
        clean = valid_input()
        self.assertEqual(sc.run_stats_checks(clean)["overall"], sc.PASS)
        clean["metrics"][0]["n"] = 10        # small sample -> warning only
        self.assertEqual(sc.run_stats_checks(clean)["overall"], sc.WARNING)

    def test_nothing_checkable_is_inconclusive_not_pass(self):
        empty = {"schema_version": sc.TEMP_SCHEMA_VERSION}
        report = sc.run_stats_checks(empty)
        self.assertEqual(report["checks"], [])
        self.assertEqual(report["overall"], sc.INCONCLUSIVE)

    def test_settings_are_applied_and_reported(self):
        report = sc.run_stats_checks(valid_input(), alpha=0.01, min_sample=200)
        self.assertEqual(report["settings"]["alpha"], 0.01)
        sample = [c for c in report["checks"] if c["check"] == "sample_size"]
        self.assertTrue(all(c["status"] == sc.WARNING for c in sample))

    def test_impossible_values_suppress_dependent_checks(self):
        doc = valid_input()
        doc["comparisons"][0]["n_b"] = 0
        checks = sc.run_stats_checks(doc)["checks"]
        names = [c["check"] for c in checks]
        self.assertIn("value_ranges", names)
        self.assertNotIn("statistical_power", names)   # n = 0 would be meaningless

    def test_records_are_included_in_internal_format(self):
        report = sc.run_stats_checks(valid_input())
        self.assertEqual(report["records"]["metrics"][0]["name"], "accuracy")
        self.assertEqual(set(report["records"]["comparisons"][0]),
                         sc.COMPARISON_FIELDS)

    def test_wrapper_equals_adapter_then_checks(self):
        raw = valid_input()
        self.assertEqual(sc.run_stats_checks(raw),
                         sc.run_checks(sc.parse_execution_result(raw)))

    def test_statistics_run_without_the_temporary_schema(self):
        # A FUTURE adapter only has to return the internal format. Built by
        # hand here - no temporary schema, no parse_execution_result.
        internal = {
            "schema_version": "some-future-adapter",
            "is_temporary_schema": False,
            "experiment_id": "hand-built",
            "metrics": [],
            "comparisons": [{"name": "c", "test": "z", "statistic": 1.96,
                             "df": None, "df2": None, "p_value": 0.05,
                             "p_operator": "=", "tails": 2,
                             "n_a": None, "n_b": None}],
        }
        report = sc.run_checks(internal)
        self.assertEqual(report["schema_version"], "some-future-adapter")
        self.assertIs(report["is_temporary_schema"], False)
        self.assertEqual(report["overall"], sc.PASS)


class TestModuleBoundaries(unittest.TestCase):

    def test_does_not_import_other_project_modules(self):
        tree = ast.parse((fx.SRC_DIR / "critic" / "stats_checks.py")
                         .read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        for other in ("multimodal", "pdf_parser", "figure_extractor",
                      "vlm_client", "critic_agent", "report"):
            self.assertFalse(any(other in n for n in names), names)

    def test_temporary_status_is_documented(self):
        doc = sc.__doc__
        self.assertIn("TEMPORARY", doc)
        self.assertIn("NOT THE MODULE 3 CONTRACT", doc)
        self.assertIn(sc.TEMP_SCHEMA_VERSION, doc)


if __name__ == "__main__":
    unittest.main()

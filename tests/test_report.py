"""
Tests for src/critic/report.py (Critic Report generation).

All inputs use the TEMPORARY test schema from stats_checks.py. Nothing here
assumes or tests compatibility with Module 3's ExecutionResult, which has not
been received.

Run from the project root:

    python -m unittest discover -s tests -v
"""

import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pdf_fixtures as fx  # also puts src/ on sys.path

from critic import report as rp
from critic import stats_checks as sc

FIXED_TIME = "2026-01-01T00:00:00+00:00"

# Written out here on purpose, NOT read from rp.NO_ISSUES_CAVEAT: if the
# constant were emptied or weakened, comparing against it would still "pass".
CAVEAT_WORDING = "does not confirm that the result is correct"

ENTRY_KEYS = {"target", "kind", "name", "test_name", "p_value",
              "confidence_interval", "effect_size", "unavailable_fields",
              "verdict", "explanation", "problems", "concerns",
              "not_checked", "checks"}
REPORT_KEYS = {"report_version", "job_id", "job_id_source", "generated_at",
               "schema_version", "is_temporary_schema", "settings", "verdict",
               "explanation", "summary", "entries", "notes"}


def doc(metrics=None, comparisons=None, experiment_id="exp-test"):
    """A TEMPORARY-schema input document."""
    data = {"schema_version": sc.TEMP_SCHEMA_VERSION,
            "metrics": metrics or [], "comparisons": comparisons or []}
    if experiment_id is not None:
        data["experiment_id"] = experiment_id
    return data


# A comparison that passes every check (64/group: Cohen's n for 80% power).
CLEAN_COMPARISON = {"name": "clean", "test": "two_sample_t", "statistic": 2.5,
                    "df": 126, "p_value": 0.014, "n_a": 64, "n_b": 64}
# A comparison where only trivial checks can run (no statistic -> no p check).
UNCHECKABLE_COMPARISON = {"name": "uncheckable", "test": "z", "p_value": 0.04}


def report_for(data, **kwargs):
    kwargs.setdefault("generated_at", FIXED_TIME)
    return rp.generate_critic_report(data, **kwargs)


def entry(report, name):
    return next(e for e in report["entries"] if e["name"] == name)


# ---------------------------------------------------------------------------
# Structure and fields
# ---------------------------------------------------------------------------

class TestReportStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = report_for(sc.EXAMPLE_EXECUTION_RESULT, job_id="job-42")

    def test_top_level_fields(self):
        self.assertEqual(set(self.report), REPORT_KEYS)
        self.assertEqual(self.report["report_version"], rp.REPORT_VERSION)
        self.assertTrue(self.report["is_temporary_schema"])

    def test_every_entry_has_every_field(self):
        for e in self.report["entries"]:
            with self.subTest(entry=e["name"]):
                self.assertEqual(set(e), ENTRY_KEYS)

    def test_one_entry_per_record_metrics_first(self):
        self.assertEqual([(e["kind"], e["name"]) for e in self.report["entries"]],
                         [("metric", "accuracy"), ("metric", "f1_score"),
                          ("comparison", "model_a_vs_baseline"),
                          ("comparison", "model_b_vs_baseline"),
                          ("comparison", "ablation_chi2")])

    def test_every_check_is_kept_once_with_its_entry(self):
        stats = sc.run_stats_checks(sc.EXAMPLE_EXECUTION_RESULT)
        attached = [c for e in self.report["entries"] for c in e["checks"]]
        self.assertEqual(len(attached), len(stats["checks"]))
        for e in self.report["entries"]:
            for check in e["checks"]:
                self.assertEqual(check["target"], e["target"])

    def test_summary_counts(self):
        summary = self.report["summary"]
        self.assertEqual(summary["entries"], 5)
        self.assertEqual(summary[rp.ISSUES_FOUND], 2)
        self.assertEqual(summary[rp.NO_ISSUES_DETECTED], 3)
        self.assertEqual(summary["checks"],
                         {"pass": 14, "warning": 4, "fail": 2, "skipped": 0})


class TestEntryFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = report_for(sc.EXAMPLE_EXECUTION_RESULT)

    def test_comparison_fields(self):
        e = entry(self.report, "model_b_vs_baseline")
        self.assertEqual(e["test_name"], "two_sample_t")
        self.assertEqual(e["p_value"]["reported"], 0.03)
        self.assertEqual(e["p_value"]["operator"], "=")
        self.assertAlmostEqual(e["p_value"]["recomputed"], 0.2376, places=4)
        self.assertEqual(e["p_value"]["consistency"], sc.FAIL)

    def test_effect_size_is_derived_and_labelled(self):
        # Hand calculation: 3.10 * sqrt(1/64 + 1/64) = 0.548
        es = entry(self.report, "model_a_vs_baseline")["effect_size"]
        self.assertAlmostEqual(es["value"], 0.548, places=3)
        self.assertEqual(es["measure"], "cohens_d")
        self.assertIn("derived", es["source"])

    def test_metric_fields(self):
        e = entry(self.report, "accuracy")
        self.assertEqual(e["confidence_interval"],
                         {"estimate": 0.91, "lower": 0.87, "upper": 0.95, "level": 0.95})
        self.assertIsNone(e["test_name"])
        self.assertIsNone(e["p_value"])

    def test_missing_fields_are_none_and_listed_not_filled(self):
        chi2 = entry(self.report, "ablation_chi2")
        self.assertIsNone(chi2["effect_size"])          # no justified conversion
        self.assertIsNone(chi2["confidence_interval"])  # schema has none
        self.assertEqual(set(chi2["unavailable_fields"]),
                         {"effect_size", "confidence_interval"})
        metric = entry(self.report, "accuracy")
        # test_name does not apply to metrics, so it is not "unavailable".
        self.assertEqual(set(metric["unavailable_fields"]), {"p_value", "effect_size"})

    def test_p_value_operator_is_kept(self):
        p = entry(self.report, "ablation_chi2")["p_value"]
        self.assertEqual((p["operator"], p["reported"]), ("<", 0.001))

    def test_problems_and_concerns_come_from_the_checks(self):
        e = entry(self.report, "f1_score")
        self.assertEqual(len(e["problems"]), 1)
        self.assertIn("outside its interval", e["problems"][0])
        self.assertIn("below the small-sample threshold", e["concerns"][0])
        self.assertIn(e["problems"][0], e["explanation"])


# ---------------------------------------------------------------------------
# Verdicts and safety against misleading conclusions
# ---------------------------------------------------------------------------

class TestVerdicts(unittest.TestCase):

    def test_verdicts_use_only_the_defined_vocabulary(self):
        report = report_for(sc.EXAMPLE_EXECUTION_RESULT)
        self.assertIn(report["verdict"], rp.ENTRY_VERDICTS + (rp.INCOMPLETE,))
        for e in report["entries"]:
            self.assertIn(e["verdict"], rp.ENTRY_VERDICTS)

    def test_failure_gives_issues_found(self):
        self.assertEqual(report_for(sc.EXAMPLE_EXECUTION_RESULT)["verdict"],
                         rp.ISSUES_FOUND)

    def test_warnings_only_give_concerns(self):
        small = dict(CLEAN_COMPARISON, name="small", df=38, n_a=20, n_b=20,
                     statistic=2.5, p_value=0.017)
        report = report_for(doc(comparisons=[small]))
        self.assertEqual(entry(report, "small")["verdict"], rp.CONCERNS)
        self.assertEqual(report["verdict"], rp.CONCERNS)

    def test_clean_result_says_no_issues_with_caveat(self):
        report = report_for(doc(comparisons=[CLEAN_COMPARISON]))
        e = entry(report, "clean")
        self.assertEqual(e["verdict"], rp.NO_ISSUES_DETECTED)
        self.assertIn(CAVEAT_WORDING, e["explanation"])
        self.assertEqual(report["verdict"], rp.NO_ISSUES_DETECTED)
        self.assertIn(CAVEAT_WORDING, report["explanation"])

    def test_only_trivial_checks_is_inconclusive_not_clean(self):
        # value_ranges passes, but the p-value cannot be checked without a
        # statistic: that must NOT read as "no issues".
        report = report_for(doc(comparisons=[UNCHECKABLE_COMPARISON]))
        e = entry(report, "uncheckable")
        self.assertEqual(e["verdict"], rp.INCONCLUSIVE)
        self.assertTrue(e["not_checked"])
        self.assertIn("Not checked", e["explanation"])
        self.assertEqual(report["verdict"], rp.INCONCLUSIVE)

    def test_metric_with_only_sample_size_is_inconclusive(self):
        report = report_for(doc(metrics=[{"name": "loss", "value": 0.2, "n": 500}]))
        self.assertEqual(entry(report, "loss")["verdict"], rp.INCONCLUSIVE)

    def test_mix_of_clean_and_unassessed_is_incomplete(self):
        report = report_for(doc(comparisons=[CLEAN_COMPARISON, UNCHECKABLE_COMPARISON]))
        self.assertEqual(report["verdict"], rp.INCOMPLETE)
        self.assertIn("uncheckable", report["explanation"])
        self.assertIn(CAVEAT_WORDING, report["explanation"])

    def test_unassessed_results_are_mentioned_even_when_issues_found(self):
        report = report_for(doc(
            metrics=[{"name": "bad_ci", "value": 2, "ci_lower": 0, "ci_upper": 1}],
            comparisons=[UNCHECKABLE_COMPARISON]))
        self.assertEqual(report["verdict"], rp.ISSUES_FOUND)
        self.assertIn("could not be assessed: uncheckable", report["explanation"])

    def test_empty_input_is_inconclusive(self):
        report = report_for(doc())
        self.assertEqual(report["entries"], [])
        self.assertEqual(report["verdict"], rp.INCONCLUSIVE)
        self.assertIn("no conclusion is drawn", report["explanation"])

    def test_notes_carry_the_standing_caveats(self):
        notes = " ".join(report_for(sc.EXAMPLE_EXECUTION_RESULT)["notes"])
        self.assertIn("TEMPORARY test schema", notes)
        self.assertIn("no compatibility with Module 3", notes)
        self.assertIn("DERIVED", notes)
        self.assertIn("not from the observed effect", notes)
        self.assertIn("do not establish whether the findings are true", notes)


class TestEffectSizeSafety(unittest.TestCase):

    def test_not_derived_from_impossible_values(self):
        bad = dict(CLEAN_COMPARISON, n_b=0)
        e = entry(report_for(doc(comparisons=[bad])), "clean")
        self.assertIsNone(e["effect_size"])
        self.assertEqual(e["verdict"], rp.ISSUES_FOUND)

    def test_paired_and_welch(self):
        paired = {"name": "p", "test": "paired_t", "statistic": 3.0, "df": 8,
                  "p_value": 0.017, "n_a": 9}
        welch = {"name": "w", "test": "welch_t", "statistic": 2.0, "df": 90,
                 "p_value": 0.049, "n_a": 50, "n_b": 50}
        report = report_for(doc(comparisons=[paired, welch]))
        self.assertAlmostEqual(entry(report, "p")["effect_size"]["value"], 1.0)
        self.assertEqual(entry(report, "p")["effect_size"]["measure"], "cohens_dz")
        self.assertTrue(entry(report, "w")["effect_size"]["approximate"])


# ---------------------------------------------------------------------------
# job_id, timestamps, inputs
# ---------------------------------------------------------------------------

class TestJobIdAndTime(unittest.TestCase):

    def test_provided_job_id(self):
        report = report_for(doc(), job_id="job-7")
        self.assertEqual((report["job_id"], report["job_id_source"]),
                         ("job-7", "provided"))

    def test_falls_back_to_experiment_id_and_says_so(self):
        report = report_for(doc(experiment_id="exp-9"))
        self.assertEqual(report["job_id"], "exp-9")
        self.assertIn("experiment_id", report["job_id_source"])

    def test_no_identifier_is_none_not_invented(self):
        report = report_for(doc(experiment_id=None))
        self.assertIsNone(report["job_id"])
        self.assertIsNone(report["job_id_source"])

    def test_generated_at(self):
        self.assertEqual(report_for(doc())["generated_at"], FIXED_TIME)
        default = rp.generate_critic_report(doc())["generated_at"]
        self.assertIsNotNone(datetime.fromisoformat(default).tzinfo)


class TestInputs(unittest.TestCase):

    def test_build_from_a_stats_report(self):
        stats = sc.run_stats_checks(doc(comparisons=[CLEAN_COMPARISON]))
        report = rp.build_critic_report(stats, generated_at=FIXED_TIME)
        self.assertEqual(report, report_for(doc(comparisons=[CLEAN_COMPARISON])))

    def test_stats_report_is_not_modified(self):
        stats = sc.run_stats_checks(sc.EXAMPLE_EXECUTION_RESULT)
        snapshot = copy.deepcopy(stats)
        rp.build_critic_report(stats)
        self.assertEqual(stats, snapshot)

    def test_rejects_things_that_are_not_stats_reports(self):
        stats = sc.run_stats_checks(doc())
        without_records = {k: v for k, v in stats.items() if k != "records"}
        bad_records = dict(stats, records={"metrics": "x", "comparisons": []})
        for label, value in (("not a dict", [1]), ("raw input", doc()),
                             ("missing records", without_records),
                             ("malformed records", bad_records)):
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    rp.build_critic_report(value)

    def test_other_input_schemas_are_rejected(self):
        with self.assertRaises(sc.SchemaError):
            rp.generate_critic_report({"run_id": "r1", "metrics": {"acc": 0.9}})

    def test_settings_reach_the_checks(self):
        report = report_for(doc(comparisons=[CLEAN_COMPARISON]), min_sample=100)
        self.assertEqual(report["settings"]["min_sample"], 100)
        self.assertEqual(report["verdict"], rp.CONCERNS)

    def test_checks_without_a_matching_record_are_kept(self):
        stats = sc.run_stats_checks(doc())
        stats["checks"].append({"check": "p_value_consistency",
                                "target": "comparison:ghost", "status": sc.FAIL,
                                "message": "orphan", "details": {}})
        report = rp.build_critic_report(stats)
        self.assertEqual(entry(report, "ghost")["problems"], ["orphan"])
        self.assertEqual(report["verdict"], rp.ISSUES_FOUND)


# ---------------------------------------------------------------------------
# Output formats and command line
# ---------------------------------------------------------------------------

class TestOutputs(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.report = report_for(sc.EXAMPLE_EXECUTION_RESULT, job_id="job-42")

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_round_trip(self):
        self.assertEqual(json.loads(rp.report_to_json(self.report)), self.report)

    def test_save_report_creates_folders(self):
        path = rp.save_report(self.report, self.dir / "out" / "report.json")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.report)

    def test_text_rendering(self):
        text = rp.format_report_text(self.report)
        self.assertIn("job: job-42", text)
        self.assertIn("Verdict: ISSUES FOUND", text)
        for e in self.report["entries"]:
            self.assertIn(e["name"], text)
        self.assertIn("cohens_d = 0.548, derived", text)
        self.assertIn("TEMPORARY test schema", text)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(fx.SRC_DIR / "critic" / "report.py"), *args],
            capture_output=True, text=True, cwd=fx.PROJECT_ROOT)

    def test_cli_writes_json(self):
        out = self.dir / "cli.json"
        result = self.run_cli("--job-id", "cli-job", "--json", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(saved["job_id"], "cli-job")
        self.assertEqual(saved["verdict"], rp.ISSUES_FOUND)

    def test_cli_rejects_other_schemas(self):
        foreign = self.dir / "foreign.json"
        foreign.write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
        result = self.run_cli(str(foreign))
        self.assertEqual(result.returncode, 1)
        self.assertIn("Nothing was guessed", result.stdout)


class TestModuleBoundaries(unittest.TestCase):

    def test_only_depends_on_stats_checks(self):
        tree = ast.parse((fx.SRC_DIR / "critic" / "report.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
        for other in ("multimodal", "pdf_parser", "figure_extractor",
                      "vlm_client", "critic_agent"):
            self.assertFalse(any(other in n for n in names), names)


if __name__ == "__main__":
    unittest.main()

"""
Tests for src/critic/critic_agent.py.

Two layers:
  * Unit tests       - the agent's own logic: settings, outcomes, error
                       handling, result shape, module boundaries.
  * End-to-end tests - the real pipeline (JSON file -> adapter -> statistical
                       checks -> Critic Report -> saved JSON / command line)
                       on the MOCK inputs in tests/fixtures/mock_execution_results/.

All inputs use Module 4's TEMPORARY test schema. Nothing here assumes or tests
compatibility with Module 3's ExecutionResult, which has not been received.

Run from the project root:

    python -m unittest discover -s tests -v
"""

import ast
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pdf_fixtures as fx  # also puts src/ on sys.path

from critic import critic_agent as ca
from critic import report as rp
from critic import stats_checks as sc

FIXED_TIME = "2026-01-01T00:00:00+00:00"
MOCK_DIR = Path(__file__).resolve().parent / "fixtures" / "mock_execution_results"

# Expected results for each mock file (also documented in the folder README).
EXPECTED = {
    "mock_completed_clean.json":         (ca.REVIEWED, rp.NO_ISSUES_DETECTED),
    "mock_completed_with_issues.json":   (ca.REVIEWED, rp.ISSUES_FOUND),
    "mock_failed_run.json":              (ca.EXPERIMENT_FAILED, None),
    "mock_missing_values.json":          (ca.REVIEWED, rp.INCONCLUSIVE),
    "mock_partially_checkable.json":     (ca.REVIEWED, rp.INCOMPLETE),
    "mock_invalid_unknown_schema.json":  (ca.INVALID_INPUT, None),
    "mock_invalid_malformed.json":       (ca.INVALID_INPUT, None),
}


def load_mock(name):
    return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))


def with_non_finite(value):
    """The clean mock with its first comparison's statistic set to `value`."""
    doc = load_mock("mock_completed_clean.json")
    doc["comparisons"][0]["statistic"] = value
    return doc


def with_duplicate_record():
    """The clean mock with its first comparison/metric repeated."""
    doc = load_mock("mock_completed_clean.json")
    kind = "comparisons" if doc.get("comparisons") else "metrics"
    doc[kind].append(copy.deepcopy(doc[kind][0]))
    return doc


def review(item, **kwargs):
    kwargs.setdefault("generated_at", FIXED_TIME)
    return ca.CriticAgent().review(item, **kwargs)


# ===========================================================================
# Unit tests
# ===========================================================================

class TestSettings(unittest.TestCase):

    def test_defaults_come_from_stats_checks(self):
        self.assertEqual(ca.CriticAgent().settings,
                         {"alpha": sc.DEFAULT_ALPHA,
                          "target_effect": sc.DEFAULT_TARGET_EFFECT,
                          "target_power": sc.DEFAULT_TARGET_POWER,
                          "min_sample": sc.DEFAULT_MIN_SAMPLE})

    def test_invalid_settings_fail_fast(self):
        bad = [dict(alpha=0), dict(alpha=1), dict(alpha=-0.1), dict(alpha="0.05"),
               dict(alpha=True), dict(target_power=1.2), dict(target_power=0),
               dict(target_effect=0), dict(target_effect=-0.5),
               dict(min_sample=0), dict(min_sample=2.5), dict(min_sample=True)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ca.CriticAgent(**kwargs)

    def test_settings_reach_the_checks(self):
        data = load_mock("mock_completed_clean.json")
        strict = ca.CriticAgent(min_sample=500).review(data)
        self.assertEqual(strict["critic_report"]["settings"]["min_sample"], 500)
        self.assertEqual(strict["verdict"], rp.CONCERNS)   # n = 64/200 < 500


class TestResultShape(unittest.TestCase):

    def test_every_outcome_has_the_same_keys(self):
        for name in EXPECTED:
            with self.subTest(mock=name):
                self.assertEqual(set(review(MOCK_DIR / name)), set(ca.RESULT_KEYS))

    def test_verdict_and_report_only_when_reviewed(self):
        for name, (outcome, _) in EXPECTED.items():
            with self.subTest(mock=name):
                result = review(MOCK_DIR / name)
                if outcome == ca.REVIEWED:
                    self.assertIsNotNone(result["critic_report"])
                    self.assertEqual(result["verdict"],
                                     result["critic_report"]["verdict"])
                else:
                    self.assertIsNone(result["verdict"])
                    self.assertIsNone(result["critic_report"])
                    self.assertTrue(result["errors"])

    def test_reviewed_result_agrees_with_its_report(self):
        result = review(load_mock("mock_completed_with_issues.json"), job_id="job-1")
        report = result["critic_report"]
        self.assertEqual(result["message"], report["explanation"])
        self.assertEqual((result["job_id"], result["job_id_source"]),
                         (report["job_id"], report["job_id_source"]))
        self.assertEqual(report["generated_at"], FIXED_TIME)
        self.assertTrue(result["is_temporary_schema"])


class TestOutcomes(unittest.TestCase):

    def test_failed_experiment_is_not_reviewed(self):
        with mock.patch.object(sc, "run_checks", wraps=sc.run_checks) as spy:
            result = review(load_mock("mock_failed_run.json"))
        spy.assert_not_called()          # no statistics on a failed run
        self.assertEqual(result["outcome"], ca.EXPERIMENT_FAILED)
        self.assertEqual(result["experiment_status"], "failed")
        self.assertIn("loss became NaN", result["message"])
        self.assertIn("no verdict is given", result["message"])
        self.assertIn("1 metric(s)", result["notes"][0])
        self.assertEqual(result["errors"][0]["stage"], "experiment")

    def test_failed_experiment_without_error_text(self):
        data = dict(load_mock("mock_failed_run.json"), metrics=[])
        del data["error"]
        result = review(data)
        self.assertEqual(result["outcome"], ca.EXPERIMENT_FAILED)
        self.assertEqual(result["errors"][0]["message"], "No error description given.")
        self.assertEqual(result["notes"], [])     # nothing was ignored

    def test_status_not_reported_is_noted_not_assumed(self):
        result = review(load_mock("mock_missing_values.json"))
        self.assertIsNone(result["experiment_status"])
        self.assertIn("status was not reported", " ".join(result["notes"]))
        completed = review(load_mock("mock_completed_clean.json"))
        self.assertEqual(completed["notes"], [])

    def test_invalid_inputs_never_raise(self):
        cases = {
            "list": [1, 2],
            "None": None,
            "number": 42,
            "wrong schema": {"schema_version": "x", "metrics": []},
            "typo in field": dict(load_mock("mock_completed_clean.json"), statuss="ok"),
            "bad status": dict(load_mock("mock_completed_clean.json"), status="crashed"),
            "NaN value": with_non_finite(float("nan")),
            "Infinity value": with_non_finite(float("inf")),
            "duplicate names": with_duplicate_record(),
        }
        for label, data in cases.items():
            with self.subTest(case=label):
                result = review(data)
                self.assertEqual(result["outcome"], ca.INVALID_INPUT)
                self.assertEqual(result["errors"][0]["stage"], "input")
                self.assertIn("nothing was reviewed", result["message"])

    def test_invalid_files_never_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            # e.g. a research PDF passed to the critic by mistake
            binary = Path(tmp) / "paper.pdf"
            binary.write_bytes(b"%PDF-1.7\n\xff\xfe\x80\x81 binary \xc3\x28")
            for label, path in (("missing file", Path(tmp) / "missing.json"),
                                ("directory", Path(tmp)),
                                ("malformed JSON", MOCK_DIR / "mock_invalid_malformed.json"),
                                ("binary file", binary)):
                with self.subTest(case=label):
                    result = review(path)
                    self.assertEqual(result["outcome"], ca.INVALID_INPUT)
                    self.assertEqual(result["source"], str(path))

    def test_internal_error_in_checks_is_reported_not_hidden(self):
        with mock.patch.object(sc, "run_checks", side_effect=RuntimeError("boom")):
            result = review(load_mock("mock_completed_clean.json"))
        self.assertEqual(result["outcome"], ca.INTERNAL_ERROR)
        self.assertIsNone(result["verdict"])
        error = result["errors"][0]
        self.assertEqual((error["stage"], error["type"], error["message"]),
                         ("checks", "RuntimeError", "boom"))
        self.assertIn("RuntimeError", error["traceback"])

    def test_internal_error_in_report_is_reported(self):
        with mock.patch.object(rp, "build_critic_report",
                               side_effect=KeyError("missing")):
            result = review(load_mock("mock_completed_clean.json"))
        self.assertEqual(result["outcome"], ca.INTERNAL_ERROR)
        self.assertEqual(result["errors"][0]["stage"], "report")


class TestInputsAndIds(unittest.TestCase):

    def test_dict_str_and_path_inputs_give_the_same_review(self):
        path = MOCK_DIR / "mock_completed_clean.json"
        from_dict = review(load_mock(path.name))
        from_str = review(str(path))
        from_path = review(path)
        self.assertEqual(from_dict["critic_report"], from_str["critic_report"])
        self.assertEqual(from_str["critic_report"], from_path["critic_report"])
        self.assertIsNone(from_dict["source"])
        self.assertEqual(from_path["source"], str(path))

    def test_job_id_rules(self):
        data = load_mock("mock_completed_clean.json")
        self.assertEqual(review(data, job_id="job-9")["job_id"], "job-9")
        fallback = review(data)
        self.assertEqual(fallback["job_id"], "mock-completed-clean")
        self.assertIn("experiment_id", fallback["job_id_source"])
        anonymous = review({k: v for k, v in data.items() if k != "experiment_id"})
        self.assertIsNone(anonymous["job_id"])

    def test_job_id_kept_for_failed_and_invalid_inputs(self):
        self.assertEqual(review(load_mock("mock_failed_run.json"))["job_id"],
                         "mock-failed-run")
        self.assertEqual(review([1], job_id="job-5")["job_id"], "job-5")

    def test_input_is_not_modified(self):
        data = load_mock("mock_completed_with_issues.json")
        snapshot = copy.deepcopy(data)
        review(data)
        self.assertEqual(data, snapshot)

    def test_same_input_same_result(self):
        data = load_mock("mock_completed_with_issues.json")
        self.assertEqual(review(data), review(data))

    def test_review_many_keeps_order(self):
        items = [MOCK_DIR / "mock_completed_clean.json", [1],
                 load_mock("mock_failed_run.json")]
        outcomes = [r["outcome"] for r in ca.CriticAgent().review_many(items)]
        self.assertEqual(outcomes, [ca.REVIEWED, ca.INVALID_INPUT,
                                    ca.EXPERIMENT_FAILED])

    def test_shortcut_function(self):
        result = ca.review_execution_result(load_mock("mock_completed_clean.json"),
                                            job_id="j", alpha=0.01)
        self.assertEqual(result["critic_report"]["settings"]["alpha"], 0.01)


class TestOutputs(unittest.TestCase):

    def test_json_round_trip_and_save(self):
        result = review(load_mock("mock_completed_with_issues.json"))
        self.assertEqual(json.loads(ca.result_to_json(result)), result)
        with tempfile.TemporaryDirectory() as tmp:
            path = ca.save_result(result, Path(tmp) / "out" / "r.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)

    def test_text_rendering(self):
        reviewed = ca.format_result_text(review(MOCK_DIR / "mock_completed_clean.json"))
        self.assertIn("outcome: REVIEWED", reviewed)
        self.assertIn("CRITIC REPORT", reviewed)
        failed = ca.format_result_text(review(MOCK_DIR / "mock_failed_run.json"))
        self.assertIn("outcome: EXPERIMENT_FAILED", failed)
        self.assertIn("verdict: (none - not reviewed)", failed)
        self.assertIn("loss became NaN", failed)


class TestModuleBoundaries(unittest.TestCase):
    """The agent coordinates; it must not pull in other modules or any LLM."""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse((fx.SRC_DIR / "critic" / "critic_agent.py")
                         .read_text(encoding="utf-8"))
        cls.imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                cls.imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                cls.imports.add(node.module)
                cls.imports.update(f"{node.module}.{a.name}" for a in node.names)

    def test_independent_of_ingestion_modules(self):
        for other in ("multimodal", "pdf_parser", "figure_extractor", "vlm_client"):
            self.assertFalse(any(other in n for n in self.imports), self.imports)

    def test_no_llm_or_network_libraries(self):
        for lib in ("openai", "anthropic", "ollama", "requests", "urllib",
                    "httpx", "transformers", "langchain"):
            self.assertFalse(any(lib in n for n in self.imports), self.imports)

    def test_uses_only_the_critic_modules(self):
        project = {n for n in self.imports
                   if n.startswith("critic") or n in ("report", "stats_checks")}
        self.assertEqual(project, {"critic", "critic.report", "critic.stats_checks",
                                   "report", "stats_checks"})


# ===========================================================================
# End-to-end integration tests (MOCK data, temporary schema)
# ===========================================================================

class TestMockData(unittest.TestCase):
    """Guards that the mock files stay clearly labelled as mock data."""

    def test_every_mock_is_listed_and_named_as_mock(self):
        files = sorted(p.name for p in MOCK_DIR.glob("*.json"))
        self.assertEqual(files, sorted(EXPECTED))
        self.assertTrue(all(name.startswith("mock_") for name in files))

    def test_valid_mocks_use_the_temporary_schema(self):
        for name in EXPECTED:
            if "invalid" in name:
                continue
            with self.subTest(mock=name):
                data = load_mock(name)
                self.assertEqual(data["schema_version"], sc.TEMP_SCHEMA_VERSION)
                self.assertTrue(data["experiment_id"].startswith("mock-"))

    def test_readme_marks_them_as_mock(self):
        text = (MOCK_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("MOCK", text)
        self.assertIn("not** Module 3 output", text)


class TestEndToEnd(unittest.TestCase):

    def test_each_mock_gives_its_expected_outcome(self):
        agent = ca.CriticAgent()
        for name, (outcome, verdict) in EXPECTED.items():
            with self.subTest(mock=name):
                result = agent.review(MOCK_DIR / name, generated_at=FIXED_TIME)
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["verdict"], verdict)

    def test_full_pipeline_details(self):
        result = review(MOCK_DIR / "mock_completed_with_issues.json")
        report = result["critic_report"]
        entries = {e["name"]: e for e in report["entries"]}

        ci = entries["f1_score"]
        self.assertEqual(ci["verdict"], rp.ISSUES_FOUND)
        self.assertIn("outside its interval", ci["problems"][0])

        comp = entries["overclaimed_comparison"]
        self.assertEqual(comp["test_name"], "two_sample_t")
        self.assertEqual(comp["p_value"]["reported"], 0.03)
        self.assertEqual(comp["p_value"]["consistency"], sc.FAIL)
        # Effect size by exact arithmetic: 1.2 * sqrt(1/64 + 1/64)
        self.assertAlmostEqual(comp["effect_size"]["value"],
                               1.2 * math.sqrt(2 / 64), places=4)
        self.assertIn("TEMPORARY test schema", " ".join(report["notes"]))

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(fx.SRC_DIR / "critic" / "critic_agent.py"), *args],
            capture_output=True, text=True, cwd=fx.PROJECT_ROOT)

    def test_cli_over_all_mocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = [str(MOCK_DIR / name) for name in EXPECTED]
            proc = self.run_cli(*inputs, "--out-dir", tmp)
            # Invalid mocks are included, so the run as a whole reports failure.
            self.assertEqual(proc.returncode, 1, proc.stderr)
            saved = {p.name: json.loads(p.read_text(encoding="utf-8"))
                     for p in Path(tmp).glob("*_critic.json")}
            self.assertEqual(len(saved), len(EXPECTED))
            for name, (outcome, verdict) in EXPECTED.items():
                with self.subTest(mock=name):
                    result = saved[name.replace(".json", "_critic.json")]
                    self.assertEqual((result["outcome"], result["verdict"]),
                                     (outcome, verdict))

    def test_cli_exit_zero_when_all_inputs_processed(self):
        # Issues found and failed experiments are still "processed".
        proc = self.run_cli(str(MOCK_DIR / "mock_completed_with_issues.json"),
                            str(MOCK_DIR / "mock_failed_run.json"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ISSUES FOUND", proc.stdout)
        self.assertIn("EXPERIMENT_FAILED", proc.stdout)

    def test_cli_built_in_example_and_job_id(self):
        proc = self.run_cli("--job-id", "job-cli")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("job: job-cli", proc.stdout)

    def test_cli_rejects_bad_options(self):
        two = [str(MOCK_DIR / "mock_completed_clean.json")] * 2
        self.assertEqual(self.run_cli(*two, "--job-id", "x").returncode, 2)
        self.assertEqual(self.run_cli("--alpha", "1.5").returncode, 2)


if __name__ == "__main__":
    unittest.main()

"""
Tests for src/multimodal/vlm_client.py.

LLaVA/Ollama is NOT required. Instead, `StubOllama` below runs a tiny local
HTTP server that speaks the same API as Ollama (`GET /api/tags`,
`POST /api/generate`) and replies with whatever a test tells it to. That way
the client's real networking code - request building, base64 encoding,
HTTP-error handling, timeouts - is exercised for real, not mocked away.

What these tests CANNOT tell you is how good LLaVA's actual descriptions are;
that needs the real model installed.

Run from the project root:

    python -m unittest discover -s tests -v
"""

import ast
import base64
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pdf_fixtures as fx  # also puts src/ on sys.path

from multimodal import vlm_client as vc


FULL_ANALYSIS = {
    "figure_summary": "Two bar charts comparing XML and LaTeX datasets.",
    "chart_type": "bar chart",
    "x_axis": "Number of figures per paper, 0 to 15+",
    "y_axis": "Fraction of papers, 0 to 0.2",
    "trends_and_patterns": "Fractions fall as the count increases.",
    "legend": "Blue = XML, orange = LaTeX",
    "important_observations": ["LaTeX has more papers with zero figures"],
    "limitations_or_uncertainty": "Small tick labels are hard to read.",
}


# ---------------------------------------------------------------------------
# A fake Ollama server
# ---------------------------------------------------------------------------

class StubOllama:
    """
    Minimal stand-in for an Ollama server, running on a free local port.

    Set the attributes before calling the client:
        tags_models      list of model names returned by /api/tags
        generate_status  HTTP status for /api/generate
        generate_body    dict (sent as JSON) or str (sent raw)
        delay            seconds to wait before answering /api/generate
    Every request body received is appended to `requests`.
    """

    def __init__(self):
        self.tags_models = ["llava:latest"]
        self.generate_status = 200
        self.generate_body = {"response": json.dumps(FULL_ANALYSIS), "done": True}
        self.delay = 0
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # keep test output quiet
                pass

            def _send(self, status, body):
                raw = body if isinstance(body, str) else json.dumps(body)
                data = raw.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": n} for n in stub.tags_models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                stub.requests.append(json.loads(self.rfile.read(length)))
                if stub.delay:
                    time.sleep(stub.delay)
                try:
                    self._send(stub.generate_status, stub.generate_body)
                except ConnectionError:
                    # The client already gave up (timeout test). Covers
                    # BrokenPipe/Reset and Windows' ConnectionAbortedError.
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def free_port_with_nothing_listening():
    """A local port that is guaranteed to refuse connections."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class StubTestCase(unittest.TestCase):
    """Gives each test a running stub server, a client for it and a temp dir."""

    def setUp(self):
        self.stub = StubOllama()
        # Model pinned so a VLM_MODEL set on the machine cannot change results.
        self.client = vc.VLMClient(model="llava", base_url=self.stub.url, timeout=5)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.image = fx.make_png(self.dir / "figure.png", 120, 80, (10, 120, 200))

    def tearDown(self):
        self.stub.close()
        self.tmp.cleanup()


# ---------------------------------------------------------------------------
# Image validation (no network involved)
# ---------------------------------------------------------------------------

class TestValidateImage(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_png_passes(self):
        png = fx.make_png(self.dir / "ok.png", 60, 40, (255, 0, 0))
        self.assertEqual(vc.validate_image(png), png)

    def test_uppercase_extension_is_accepted(self):
        png = fx.make_png(self.dir / "OK.PNG", 60, 40, (255, 0, 0))
        self.assertEqual(vc.validate_image(png), png)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            vc.validate_image(self.dir / "nope.png")

    def test_directory(self):
        with self.assertRaises(FileNotFoundError):
            vc.validate_image(self.dir)

    def test_unsupported_extension(self):
        for name in ("chart.gif", "chart.bmp", "chart.pdf", "chart.txt"):
            with self.subTest(name=name):
                path = self.dir / name
                path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
                with self.assertRaisesRegex(ValueError, "Unsupported image format"):
                    vc.validate_image(path)

    def test_renamed_non_image_is_rejected(self):
        fake = self.dir / "fake.png"
        fake.write_text("this is text, not a PNG", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a valid PNG or JPEG"):
            vc.validate_image(fake)

    def test_empty_file(self):
        empty = self.dir / "empty.jpg"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            vc.validate_image(empty)

    def test_too_large(self):
        big = fx.make_png(self.dir / "big.png", 60, 40, (0, 0, 0))
        with mock.patch.object(vc, "MAX_IMAGE_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "too large"):
                vc.validate_image(big)

    def test_real_extracted_figures_pass(self):
        folder = fx.PROJECT_ROOT / "sample_pdfs" / "extracted_figures"
        images = sorted(folder.glob("*.*")) if folder.exists() else []
        if not images:
            self.skipTest("run figure_extractor.py first to create extracted_figures/")
        for image in images:          # includes both .png and .jpeg files
            with self.subTest(image=image.name):
                vc.validate_image(image)


# ---------------------------------------------------------------------------
# Prompt and response parsing (no network involved)
# ---------------------------------------------------------------------------

class TestPrompt(unittest.TestCase):

    def test_caption_is_included_as_context(self):
        prompt = vc.build_prompt("Figure 2: Distributions of figures.")
        self.assertIn('Caption: "Figure 2: Distributions of figures."', prompt)

    def test_no_caption_is_stated(self):
        self.assertIn("No caption is available", vc.build_prompt(None))

    def test_every_field_is_requested_and_guessing_discouraged(self):
        prompt = vc.build_prompt("x")
        for field in vc.ANALYSIS_FIELDS:
            self.assertIn(f'"{field}"', prompt)
        self.assertIn("Do not invent numbers", prompt)


class TestParseAnalysis(unittest.TestCase):

    def test_complete_reply_is_ok(self):
        analysis, missing, status = vc.parse_analysis(json.dumps(FULL_ANALYSIS))
        self.assertEqual(status, "ok")
        self.assertEqual(missing, [])
        self.assertEqual(analysis, FULL_ANALYSIS)

    def test_missing_fields_are_none_and_listed(self):
        reply = {"figure_summary": "A diagram.", "chart_type": "diagram"}
        analysis, missing, status = vc.parse_analysis(json.dumps(reply))
        self.assertEqual(status, "partial")
        self.assertEqual(analysis["chart_type"], "diagram")
        self.assertIsNone(analysis["x_axis"])
        self.assertEqual(len(missing), 6)
        self.assertIn("legend", missing)

    def test_blank_values_count_as_missing(self):
        reply = dict(FULL_ANALYSIS, legend="  ", important_observations=[])
        _, missing, status = vc.parse_analysis(json.dumps(reply))
        self.assertEqual(status, "partial")
        self.assertEqual(set(missing), {"legend", "important_observations"})

    def test_alternative_key_names_are_mapped(self):
        reply = {"Summary": "s", "Chart Type": "line plot", "X-Axis": "time",
                 "Y-Axis Information": "loss", "trends": "falls",
                 "Legend Information": "none", "Observations": ["a"],
                 "Limitations": "blurry"}
        analysis, missing, status = vc.parse_analysis(json.dumps(reply))
        self.assertEqual(status, "ok", missing)
        self.assertEqual(analysis["x_axis"], "time")
        self.assertEqual(analysis["y_axis"], "loss")

    def test_json_inside_code_fence_or_prose(self):
        body = json.dumps(FULL_ANALYSIS)
        for text in (f"```json\n{body}\n```", f"Here is the analysis: {body}"):
            with self.subTest(text=text[:20]):
                _, _, status = vc.parse_analysis(text)
                self.assertEqual(status, "ok")

    def test_plain_text_reply_is_unstructured(self):
        analysis, missing, status = vc.parse_analysis("It is a bar chart of results.")
        self.assertEqual(status, "unstructured")
        self.assertTrue(all(value is None for value in analysis.values()))
        self.assertEqual(missing, vc.ANALYSIS_FIELDS)

    def test_unknown_keys_are_ignored(self):
        reply = dict(FULL_ANALYSIS, colour_scheme="blue/orange")
        analysis, _, _ = vc.parse_analysis(json.dumps(reply))
        self.assertEqual(set(analysis), set(vc.ANALYSIS_FIELDS))


# ---------------------------------------------------------------------------
# Talking to the (stub) server
# ---------------------------------------------------------------------------

class TestAnalyzeFigure(StubTestCase):

    def test_successful_analysis(self):
        result = self.client.analyze_figure(self.image, caption="Figure 2: Bars.")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["analysis"]["chart_type"], "bar chart")
        self.assertEqual(result["caption"], "Figure 2: Bars.")
        self.assertEqual(result["model"], "llava")
        self.assertEqual(result["image_path"], str(self.image))
        self.assertEqual(result["raw_response"], json.dumps(FULL_ANALYSIS))
        self.assertGreaterEqual(result["duration_seconds"], 0)
        # A real-mode result must be labelled as real, never as mock.
        self.assertIs(result["is_mock"], False)
        self.assertEqual(result["source"], "ollama")
        self.assertIsNone(result["disclaimer"])

    def test_request_sent_to_ollama_is_correct(self):
        self.client.analyze_figure(self.image, caption="Figure 2: Bars.")
        sent = self.stub.requests[-1]
        self.assertEqual(sent["model"], "llava")
        self.assertIs(sent["stream"], False)
        self.assertEqual(sent["format"], "json")
        self.assertEqual(sent["options"]["temperature"], 0)
        self.assertIn("Figure 2: Bars.", sent["prompt"])
        # The image must arrive byte-for-byte intact.
        self.assertEqual(base64.b64decode(sent["images"][0]), self.image.read_bytes())

    def test_partial_reply(self):
        self.stub.generate_body = {"response": json.dumps({"chart_type": "diagram"})}
        result = self.client.analyze_figure(self.image)
        self.assertEqual(result["status"], "partial")
        self.assertIn("x_axis", result["missing_fields"])

    def test_model_not_installed(self):
        self.stub.generate_status = 404
        self.stub.generate_body = {"error": "model 'llava' not found, try pulling it first"}
        with self.assertRaisesRegex(vc.VLMModelError, "ollama pull llava"):
            self.client.analyze_figure(self.image)

    def test_server_side_error(self):
        self.stub.generate_status = 500
        self.stub.generate_body = {"error": "out of memory"}
        with self.assertRaisesRegex(vc.VLMModelError, "out of memory"):
            self.client.analyze_figure(self.image)

    def test_error_inside_a_200_reply(self):
        self.stub.generate_body = {"error": "image could not be decoded"}
        with self.assertRaisesRegex(vc.VLMModelError, "could not be decoded"):
            self.client.analyze_figure(self.image)

    def test_reply_that_is_not_json(self):
        self.stub.generate_body = "<html>proxy error</html>"
        with self.assertRaises(vc.VLMResponseError):
            self.client.analyze_figure(self.image)

    def test_empty_model_response(self):
        self.stub.generate_body = {"response": "   ", "done": True}
        with self.assertRaisesRegex(vc.VLMResponseError, "empty response"):
            self.client.analyze_figure(self.image)

    def test_timeout(self):
        self.stub.delay = 2
        client = vc.VLMClient(base_url=self.stub.url, timeout=0.5)
        with self.assertRaises(vc.VLMConnectionError):
            client.analyze_figure(self.image)

    def test_bad_image_fails_before_any_request(self):
        with self.assertRaises(FileNotFoundError):
            self.client.analyze_figure(self.dir / "missing.png")
        self.assertEqual(self.stub.requests, [])


class TestNoServer(unittest.TestCase):
    """Real behaviour when nothing is listening (the situation on this PC)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image = fx.make_png(Path(self.tmp.name) / "f.png", 60, 40, (0, 0, 0))
        port = free_port_with_nothing_listening()
        self.client = vc.VLMClient(base_url=f"http://127.0.0.1:{port}", timeout=5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyze_raises_connection_error_with_hint(self):
        with self.assertRaisesRegex(vc.VLMConnectionError, "ollama serve"):
            self.client.analyze_figure(self.image)

    def test_check_ready_reports_without_raising(self):
        status = self.client.check_ready()
        self.assertFalse(status["server_reachable"])
        self.assertFalse(status["model_available"])
        self.assertIn("ollama serve", status["message"])


class TestReadiness(StubTestCase):

    def test_ready_when_llava_latest_installed(self):
        status = self.client.check_ready()
        self.assertTrue(status["server_reachable"])
        self.assertTrue(status["model_available"])
        self.assertEqual(status["installed_models"], ["llava:latest"])

    def test_untagged_name_does_not_match_other_tags(self):
        # Ollama resolves "llava" to "llava:latest", so llava:13b alone is not enough.
        self.stub.tags_models = ["llava:13b"]
        status = self.client.check_ready()
        self.assertTrue(status["server_reachable"])
        self.assertFalse(status["model_available"])
        self.assertIn("ollama pull llava", status["message"])

    def test_explicit_tag_matches(self):
        self.stub.tags_models = ["llava:13b"]
        client = vc.VLMClient(model="llava:13b", base_url=self.stub.url)
        self.assertTrue(client.check_ready()["model_available"])


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

class TestAnalyzeFigures(StubTestCase):

    def test_one_bad_record_does_not_stop_the_batch(self):
        records = [
            {"file_path": str(self.image), "caption": "Figure 1: ok."},
            {"file_path": None, "caption": "Figure 2: not located."},
            {"file_path": str(self.dir / "gone.png"), "caption": "Figure 3: missing."},
            {"file_path": str(self.image)},
        ]
        results = vc.analyze_figures(records, client=self.client)
        self.assertEqual([r["status"] for r in results], ["ok", "skipped", "error", "ok"])
        self.assertIn("not found", results[2]["error"])
        self.assertIs(results[1]["record"], records[1])
        self.assertTrue(all(r["is_mock"] is False for r in results))
        self.assertEqual(self.stub.requests[0]["prompt"].count("Figure 1: ok."), 1)

    def test_unreachable_server_is_tried_only_once(self):
        port = free_port_with_nothing_listening()
        client = vc.VLMClient(base_url=f"http://127.0.0.1:{port}", timeout=5)
        records = [{"file_path": str(self.image)} for _ in range(3)]
        with mock.patch.object(client, "analyze_figure",
                               wraps=client.analyze_figure) as spy:
            results = vc.analyze_figures(records, client=client)
        self.assertEqual(spy.call_count, 1)
        self.assertTrue(all(r["status"] == "error" for r in results))
        self.assertEqual(len({r["error"] for r in results}), 1)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def _all_text_values(analysis):
    """Every string value in an analysis dict, flattening lists."""
    values = []
    for value in analysis.values():
        values.extend(value if isinstance(value, list) else [value])
    return values


class MockTestCase(unittest.TestCase):
    """
    Temp dir + image for each test, and a guard that FAILS the test if any
    code path tries to open a network connection - mock mode must never do so.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.image = fx.make_png(self.dir / "chart.png", 120, 80, (200, 80, 10))
        guard = mock.patch.object(
            vc.urllib.request, "urlopen",
            side_effect=AssertionError("mock mode tried to use the network"))
        self.network_guard = guard.start()
        self.addCleanup(guard.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, scenario="ok"):
        return vc.MockVLMClient(scenario=scenario, model="llava")


class TestMockSuccess(MockTestCase):

    def test_ok_scenario_returns_all_fields(self):
        result = self.client().analyze_figure(self.image, caption="Figure 2: Bars.")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["missing_fields"], [])
        self.assertEqual(set(result["analysis"]), set(vc.ANALYSIS_FIELDS))
        self.assertEqual(result["caption"], "Figure 2: Bars.")
        self.assertEqual(result["image_path"], str(self.image))

    def test_result_is_labelled_mock_in_every_way(self):
        result = self.client().analyze_figure(self.image)
        self.assertIs(result["is_mock"], True)
        self.assertEqual(result["source"], "mock")
        self.assertEqual(result["disclaimer"], vc.MOCK_DISCLAIMER)
        for value in _all_text_values(result["analysis"]):
            self.assertTrue(value.startswith(vc.MOCK_PREFIX), value)

    def test_mock_never_touches_the_network(self):
        client = self.client()
        client.analyze_figure(self.image)
        client.check_ready()
        client.list_models()
        self.network_guard.assert_not_called()

    def test_prompt_built_by_the_real_code_is_recorded(self):
        client = self.client()
        client.analyze_figure(self.image, caption="Figure 7: Loss curve.")
        self.assertEqual(len(client.calls), 1)
        self.assertIn('Caption: "Figure 7: Loss curve."', client.calls[0]["prompt"])
        self.assertEqual(client.calls[0]["image_path"], str(self.image))

    def test_partial_scenario(self):
        result = self.client("partial").analyze_figure(self.image)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["missing_fields"]), 6)
        self.assertIsNone(result["analysis"]["x_axis"])
        self.assertTrue(result["analysis"]["chart_type"].startswith(vc.MOCK_PREFIX))

    def test_unstructured_scenario(self):
        result = self.client("unstructured").analyze_figure(self.image)
        self.assertEqual(result["status"], "unstructured")
        self.assertTrue(result["raw_response"].startswith(vc.MOCK_PREFIX))
        self.assertIs(result["is_mock"], True)

    def test_check_ready_says_it_is_a_mock(self):
        status = self.client().check_ready()
        self.assertEqual(status["mode"], "mock")
        self.assertTrue(status["model_available"])
        self.assertIn(vc.MOCK_PREFIX, status["message"])
        self.assertIn("no AI model", status["message"])

    def test_works_on_a_real_extracted_figure(self):
        folder = fx.PROJECT_ROOT / "sample_pdfs" / "extracted_figures"
        images = sorted(folder.glob("*.*")) if folder.exists() else []
        if not images:
            self.skipTest("run figure_extractor.py first to create extracted_figures/")
        result = self.client().analyze_figure(images[0])
        self.assertEqual(result["status"], "ok")
        self.assertIs(result["is_mock"], True)


class TestMockErrors(MockTestCase):

    def test_missing_image_is_still_rejected(self):
        client = self.client()
        with self.assertRaises(FileNotFoundError):
            client.analyze_figure(self.dir / "missing.png")
        self.assertEqual(client.calls, [])   # failed before "calling the model"

    def test_fake_image_is_still_rejected(self):
        fake = self.dir / "fake.png"
        fake.write_text("not an image", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.client().analyze_figure(fake)

    def test_empty_reply(self):
        with self.assertRaisesRegex(vc.VLMResponseError, "empty response"):
            self.client("empty").analyze_figure(self.image)

    def test_model_missing(self):
        client = self.client("model_missing")
        with self.assertRaisesRegex(vc.VLMModelError, "ollama pull llava"):
            client.analyze_figure(self.image)
        status = client.check_ready()
        self.assertFalse(status["model_available"])
        self.assertEqual(status["mode"], "mock")

    def test_connection_error(self):
        client = self.client("connection_error")
        with self.assertRaisesRegex(vc.VLMConnectionError, "Simulated"):
            client.analyze_figure(self.image)
        status = client.check_ready()
        self.assertFalse(status["server_reachable"])
        self.assertFalse(status["model_available"])

    def test_simulated_errors_are_labelled_mock(self):
        for scenario, error in (("model_missing", vc.VLMModelError),
                                ("connection_error", vc.VLMConnectionError)):
            with self.subTest(scenario=scenario):
                with self.assertRaises(error) as caught:
                    self.client(scenario).analyze_figure(self.image)
                self.assertIn(vc.MOCK_PREFIX, str(caught.exception))

    def test_unknown_scenario(self):
        with self.assertRaisesRegex(ValueError, "Unknown mock scenario"):
            vc.MockVLMClient(scenario="explode")


class TestMockBatch(MockTestCase):

    def test_batch_in_mock_mode(self):
        records = [{"file_path": str(self.image), "caption": "Figure 1: a."},
                   {"file_path": None},
                   {"file_path": str(self.image)}]
        results = vc.analyze_figures(records, client=self.client())
        self.assertEqual([r["status"] for r in results], ["ok", "skipped", "ok"])
        self.assertTrue(all(r["is_mock"] is True for r in results))
        self.assertTrue(results[0]["result"]["is_mock"])

    def test_batch_stops_after_simulated_connection_error(self):
        client = self.client("connection_error")
        records = [{"file_path": str(self.image)} for _ in range(3)]
        results = vc.analyze_figures(records, client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(all(r["status"] == "error" for r in results))


class TestCreateClient(unittest.TestCase):

    def test_explicit_choice(self):
        self.assertIsInstance(vc.create_client(mock=True), vc.MockVLMClient)
        real = vc.create_client(mock=False)
        self.assertIs(type(real), vc.VLMClient)
        self.assertFalse(real.is_mock)

    def test_vlm_mock_environment_variable(self):
        for value, expect_mock in (("1", True), ("true", True), ("YES", True),
                                   ("0", False), ("", False)):
            with self.subTest(VLM_MOCK=value):
                with mock.patch.dict("os.environ", {"VLM_MOCK": value}):
                    self.assertEqual(vc.create_client().is_mock, expect_mock)

    def test_settings_are_passed_through(self):
        client = vc.create_client(mock=True, scenario="partial",
                                  model="llava:13b", base_url="http://h:1/")
        self.assertEqual((client.scenario, client.model, client.base_url),
                         ("partial", "llava:13b", "http://h:1"))


class TestMockMatchesRealShape(StubTestCase):
    """Code written against mock results must work unchanged on real ones."""

    def test_same_result_keys_and_fields(self):
        real = self.client.analyze_figure(self.image)
        fake = vc.MockVLMClient(model="llava").analyze_figure(self.image)
        self.assertEqual(set(real), set(fake))
        self.assertEqual(set(real["analysis"]), set(fake["analysis"]))
        self.assertNotEqual(real["source"], fake["source"])


# ---------------------------------------------------------------------------
# Configuration and module boundaries
# ---------------------------------------------------------------------------

class TestConfiguration(unittest.TestCase):

    def test_default_url_and_model(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            client = vc.VLMClient()
        self.assertEqual(client.base_url, "http://localhost:11434")
        self.assertEqual(client.model, "llava")

    def test_model_from_environment(self):
        with mock.patch.dict("os.environ", {"VLM_MODEL": "llava:13b"}):
            self.assertEqual(vc.VLMClient().model, "llava:13b")

    def test_explicit_model_wins_over_environment(self):
        with mock.patch.dict("os.environ", {"VLM_MODEL": "llava:13b"}):
            self.assertEqual(vc.VLMClient(model="bakllava").model, "bakllava")

    def test_command_line_options(self):
        args = vc._parse_args(["fig.png", "--caption", "Figure 2: x",
                               "--model", "llava:13b", "--url", "http://h:1",
                               "--mock", "--scenario", "partial"])
        self.assertEqual((args.image, args.caption, args.model, args.url),
                         ("fig.png", "Figure 2: x", "llava:13b", "http://h:1"))
        self.assertTrue(args.mock)
        self.assertEqual(args.scenario, "partial")

    def test_ollama_host_without_scheme(self):
        with mock.patch.dict("os.environ", {"OLLAMA_HOST": "127.0.0.1:9999"}):
            self.assertEqual(vc.VLMClient().base_url, "http://127.0.0.1:9999")

    def test_explicit_url_wins_and_trailing_slash_removed(self):
        with mock.patch.dict("os.environ", {"OLLAMA_HOST": "other:1"}):
            client = vc.VLMClient(base_url="http://host:1234/")
        self.assertEqual(client.base_url, "http://host:1234")

    def test_vlm_client_does_not_import_other_module4_files(self):
        tree = ast.parse((fx.SRC_DIR / "multimodal" / "vlm_client.py")
                         .read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        for other in ("pdf_parser", "figure_extractor"):
            self.assertFalse(any(other in n for n in names), names)


if __name__ == "__main__":
    unittest.main()


# Module 4 – Multimodal Ingestion and Statistical Critic


## Responsibilities


1. Extract text from research PDFs.
2. Extract charts and figures.
3. Analyze charts using LLaVA.
4. Perform statistical analysis.
5. Generate Critic Reports.

## Ingestion components (text vs visual split)

The ingestion stage is split across two files with clearly separated jobs, so
each kind of content is handled where it can be handled reliably:

### `src/multimodal/pdf_parser.py` — text and text-tables
- `extract_text_from_pdf(pdf_path)` — plain page text, one record per page.
- `extract_tables_from_pdf(pdf_path)` — **text-based tables** (rows and cells)
  detected with PyMuPDF's `find_tables()`. Only works on tables whose numbers
  and labels are selectable text.

### `src/multimodal/figure_extractor.py` — images and figures
- `extract_embedded_images(pdf_path)` — raw embedded bitmaps, one record per
  placement; repeats are kept and marked with `duplicate_of`.
- `extract_figure_regions(pdf_path)` — caption-anchored **figure** crops,
  rendered so that vector charts, arrows, labels and composite figures are
  captured whole. (Figures only — tables are not cropped here.)
- `detect_tables(pdf_path)` — triages every `Table N:` caption:
  - **text-based** → routed to `pdf_parser.extract_tables_from_pdf`;
  - **image-based** → flagged `requires_ocr` and its region saved to
    `sample_pdfs/tables_requiring_ocr/` for OCR / manual review (never dropped);
  - **unresolved** → recorded for manual review.
- `extract_all(pdf_path)` — runs the three passes above and returns
  `{embedded_images, figure_regions, tables, pdf_path}`.

### `src/multimodal/vlm_client.py` — figure analysis with LLaVA
- `VLMClient().analyze_figure(image_path, caption=None)` — sends a PNG/JPEG
  figure (plus caption as context) to LLaVA via Ollama's local REST API and
  returns a structured analysis: `figure_summary`, `chart_type`, `x_axis`,
  `y_axis`, `trends_and_patterns`, `legend`, `important_observations`,
  `limitations_or_uncertainty`. Fields the model does not provide are `None`
  (never invented); `status` is `ok`, `partial` or `unstructured`.
- `VLMClient().check_ready()` — reports whether Ollama is reachable and the
  model is installed, with the command to fix it if not.
- `analyze_figures(records)` — batch helper for `figure_regions` records; one
  failing figure never stops the batch.
- **Mock mode** — `MockVLMClient()` (or `--mock`, or `VLM_MOCK=1`) runs the
  same workflow with **no server and no model**, returning placeholder
  results. Mock results are labelled `is_mock: True`, `source: "mock"`, carry a
  `disclaimer`, and every value starts with `[MOCK]` — they are **not** AI
  analysis. Scenarios (`ok`, `partial`, `unstructured`, `empty`,
  `model_missing`, `connection_error`) let downstream code test error paths.
- Configurable: model via `model=` / `VLM_MODEL` / `--model`; server via
  `base_url=` / `OLLAMA_HOST` / `--url`.
- Uses only the Python standard library; does not import the other modules.
- **Real mode requires** (not installed automatically): Ollama
  (https://ollama.com) and a vision model, e.g. `ollama pull llava`.

```
python src/multimodal/vlm_client.py IMAGE --caption "Figure 2: ..." --mock
```

### `src/critic/stats_checks.py` — statistical checks

> ⚠️ **Temporary input schema — NOT the Module 3 contract.** Module 3's
> `ExecutionResult` schema has not been received (it is not in the repo, any
> branch, issues, PRs or wiki). `stats_checks.py` reads only a Module 4
> **temporary test schema** (`"schema_version":
> "module4-temporary-test-schema-v0"`, documented in the file) and rejects
> everything else. **No compatibility with Module 3 is claimed.**

- Layers: `parse_execution_result()` (the **only** schema-aware adapter) →
  internal format → `run_checks()` (statistics, never sees the input schema).
  When Module 3's schema arrives, only the adapter changes.
- **Reported-value consistency:** recomputes p from t / z / F / chi-square
  statistics (respecting the reported rounding and `<` / `>` reports), checks
  degrees of freedom against sample sizes, confidence intervals against their
  estimate, and flags impossible values.
- **Sample size and power:** small-sample warning (heuristic, default 30),
  power to detect a *target* effect (default Cohen's d = 0.5), minimum
  detectable effect and required n. Post-hoc "observed power" is deliberately
  not used.
- Each check returns `pass` / `warning` / `fail` / `skipped`; the report's
  `overall` is the worst of these, or `inconclusive` if nothing could be checked.

```
python src/critic/stats_checks.py                      # built-in example
python src/critic/stats_checks.py my_result.json       # temporary-schema file
```

### `src/critic/report.py` — Critic Report

Turns the `stats_checks` report into a structured **Critic Report**: one entry
per metric/comparison with `test_name`, `p_value` (reported and recomputed),
`confidence_interval`, `effect_size`, `verdict`, `explanation` and the
individual check results, plus an overall job `verdict`, `job_id` and `notes`.

- Consumes the stats report, never the raw input, so it inherits the
  temporary-schema adapter approach (still **not** Module 3's schema).
- **No misleading conclusions:** verdicts are `issues_found`, `concerns`,
  `no_issues_detected`, `inconclusive` (and `incomplete` for a job mixing
  checked and unchecked results) — never "valid" or "significant".
  `no_issues_detected` always carries a caveat; results where only trivial
  checks ran are `inconclusive`; anything not checked is listed.
- Missing values are `None` and listed in `unavailable_fields`; nothing is
  estimated to fill gaps. `effect_size` is **derived** Cohen's d / d_z from t
  statistics only, and labelled as derived.
- `job_id`: from `--job-id`/argument; otherwise the input's `experiment_id`,
  with `job_id_source` saying so; otherwise `None`.

```
python src/critic/report.py --job-id job-42                  # built-in example
python src/critic/report.py my_result.json --json out.json   # save JSON report
```

Run either file directly to test it:

```
python src/multimodal/pdf_parser.py        # text + text-tables
python src/multimodal/figure_extractor.py  # images + figure crops + table triage
```

Run the unit tests (standard-library `unittest`, no extra packages):

```
python -m unittest discover -s tests -v
```

Tests build small synthetic PDFs with known contents and write all output to
temporary folders, so they never touch `sample_pdfs/`.

Outputs land in `sample_pdfs/`: `extracted_text.txt`, `extracted_tables.txt`,
`extracted_figures/`, `extracted_figure_regions/`, and (only if needed)
`tables_requiring_ocr/`.

**Known limitation:** this ingestion stage uses no LLM or CV model, so figure
crop boundaries are best-effort (flagged `detected` / `uncertain`) and
PyMuPDF's table finder can merge complex multi-row headers into single cells.
Chart *analysis* (LLaVA) and statistical critique are later steps.

## Technologies

- Python
- PyMuPDF
- Pillow
- LLaVA
- SciPy
- Statsmodels
- NumPy

# Mock ExecutionResult inputs (Module 4 tests)

**These are MOCK files.** They are hand-written by Module 4 for testing and
use Module 4's **temporary test schema** (`module4-temporary-test-schema-v0`,
defined in `src/critic/stats_checks.py`). They are **not** Module 3 output,
and nothing here implies compatibility with Module 3's real `ExecutionResult`
schema, which has not been received yet.

| File | Scenario | Expected agent outcome / verdict |
|---|---|---|
| `mock_completed_clean.json` | consistent values, 64 per group | `reviewed` / `no_issues_detected` |
| `mock_completed_with_issues.json` | estimate outside its CI; p = 0.03 reported for t = 1.20 | `reviewed` / `issues_found` |
| `mock_failed_run.json` | `"status": "failed"` with an error message | `experiment_failed` / no verdict |
| `mock_missing_values.json` | no statistic, no CI, status omitted | `reviewed` / `inconclusive` |
| `mock_partially_checkable.json` | one checkable comparison, one not | `reviewed` / `incomplete` |
| `mock_invalid_unknown_schema.json` | unsupported `schema_version` | `invalid_input` / no verdict |
| `mock_invalid_malformed.json` | broken JSON (deliberately) | `invalid_input` / no verdict |

Replace or extend these with real Module 3 samples once its schema is known.

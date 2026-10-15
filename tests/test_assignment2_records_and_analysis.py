import json
import tempfile
import unittest
from pathlib import Path

from assignment2.analysis import load_records, paired_differences, repeated_run_variability, summary_rows
from assignment2.storage import atomic_json_write, matching_completed_record, next_attempt_path


def record(model, agent, progression, status="completed", repeat_index=0, variability=False):
    spec = {
        "model": model,
        "benchmark": "babyai",
        "agent": agent,
        "task": "task-a",
        "seed": 7,
        "repeat_index": repeat_index,
    }
    if variability:
        spec["variability_subset"] = "fixed"
    return {
        "status": status,
        "data_kind": "final",
        "config_fingerprint": "same",
        "spec": spec,
        "outcome": {"progression": progression, "success": progression == 1, "steps": 4, "invalid_actions": 1},
        "usage": {
            "episode_estimated_cost_usd": 0.02,
            "calls": [{"role": "actor", "input_tokens": None, "output_tokens": None}],
        },
        "elapsed_seconds": 2.0,
    }


class RecordAndAnalysisTests(unittest.TestCase):
    def test_resume_requires_matching_completed_record_and_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            episodes = Path(directory)
            first = next_attempt_path(episodes, "episode")
            atomic_json_write(first, {"status": "incomplete", "config_fingerprint": "one"})
            self.assertIsNone(matching_completed_record(episodes, "episode", "one"))
            second = next_attempt_path(episodes, "episode")
            atomic_json_write(second, {"status": "completed", "config_fingerprint": "one"})
            self.assertEqual(matching_completed_record(episodes, "episode", "one"), second)
            self.assertIsNone(matching_completed_record(episodes, "episode", "different"))
            self.assertEqual(next_attempt_path(episodes, "episode").name, "episode_attempt-03.json")

    def test_summary_excludes_incomplete_and_pairs_only_matched_cells(self):
        records = [
            record("model-a", "naive", 0.2),
            record("model-a", "planning", 0.6),
            record("model-a", "reflective_planning", 0.7),
            record("model-a", "naive", 0.0, status="incomplete"),
        ]
        rows = summary_rows(records)
        naive = next(row for row in rows if row["agent"] == "naive")
        self.assertEqual(naive["n_completed"], 1)
        self.assertEqual(naive["mean_progression"], 0.2)
        self.assertEqual(naive["input_tokens"], 0)
        pairs = paired_differences(records, "planning", "naive", bootstrap_samples=20)
        self.assertAlmostEqual(pairs[0]["mean_difference"], 0.4)

    def test_repeated_variability_is_separate_from_task_aggregation(self):
        records = [
            record("model-a", "planning", 0.2, repeat_index=1, variability=True),
            record("model-a", "planning", 0.8, repeat_index=2, variability=True),
        ]
        rows = repeated_run_variability(records)
        self.assertEqual(rows[0]["n_repeats"], 2)
        self.assertAlmostEqual(rows[0]["mean_progression"], 0.5)
        self.assertEqual(summary_rows(records), [])

    def test_analysis_refuses_synthetic_or_mixed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"
            episodes = run_directory / "episodes"
            episodes.mkdir(parents=True)
            synthetic = record("model-a", "naive", 0.1)
            synthetic["data_kind"] = "synthetic"
            (episodes / "one.json").write_text(json.dumps(synthetic), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                load_records([run_directory])

"""Offline checks for exp007 score integration; no model training."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_gsu_integrated_guidance as ig


class IntegratedScoreTests(unittest.TestCase):
    def test_historical_prior_is_finite_and_layer_complete(self):
        prior, meta = ig.load_historical_damage_prior()
        self.assertEqual(set(prior), set(range(28)))
        self.assertTrue(all(0.0 < value <= 1.0 for value in prior.values()))
        self.assertTrue(meta["mlp_only"])
        self.assertEqual(meta["attention_prior"], 1.0)
        self.assertEqual(len(meta["source_files"]), 4)

    def test_mlp_is_scaled_but_attention_is_neutral(self):
        prior = {layer: (0.1 if layer == 0 else 1.0) for layer in range(28)}
        rows = [
            {"component": "L00.mlp.up_proj", "shared_score_S": 1.0},
            {"component": "L27.mlp.down_proj", "shared_score_S": 1.0},
            {"component": "L00.self_attn.q_proj", "shared_score_S": 1.0},
        ]
        integrated, _ = ig.component_integrated_score_rows_for_test(rows, prior)
        indexed = {row["component"]: row for row in integrated}
        self.assertEqual(indexed["L00.mlp.up_proj"]["integrated_score_S"], 0.1)
        self.assertEqual(indexed["L27.mlp.down_proj"]["integrated_score_S"], 1.0)
        self.assertEqual(indexed["L00.self_attn.q_proj"]["integrated_score_S"], 1.0)
        self.assertEqual(indexed["L00.self_attn.q_proj"]["historical_damage_prior"], 1.0)

    def test_all_variants_use_the_expected_mode(self):
        rows = [
            {"component": "L00.mlp.up_proj", "shared_score_S": 1.0},
            {"component": "L27.mlp.down_proj", "shared_score_S": 0.5},
            {"component": "L00.self_attn.q_proj", "shared_score_S": 0.2},
            {"component": "L27.self_attn.o_proj", "shared_score_S": 0.1},
        ]
        for variant in ig.VARIANTS:
            with self.subTest(variant=variant):
                guide = ig.integrated_component_guide(rows, variant)
                self.assertEqual(guide["variant"], variant)
                self.assertEqual(
                    guide["applied"], variant not in ("B0", "B1")
                )

    def test_historical_files_are_inputs_not_training_targets(self):
        self.assertTrue(all(path.is_file() for path in ig.HISTORICAL_FILES))
        self.assertEqual(ig.METHOD, "GSU-v7-integrated-damage-gradient-path")

    def test_missing_historical_file_fails_loudly(self):
        with mock.patch.object(ig, "HISTORICAL_FILES", (Path("missing.json"),) * 4):
            with self.assertRaises(FileNotFoundError):
                ig.load_historical_damage_prior()


class IntegratedComparisonTests(unittest.TestCase):
    def make_runs(self, folder, variants=ig.VARIANTS):
        paths = []
        for variant in variants:
            path = Path(folder) / variant
            path.mkdir()
            config = {key: "same" for key in (
                "model", "seed", "lr", "updates", "lora", "optimizer", "feedback",
                "input_sha256", "source_sha256", "model_metadata_sha256", "model_files",
            )}
            config["variant"] = variant
            ig.base.write_json(path / "config.json", config)
            ig.base.write_json(path / "resolved_config.json", {
                "base_scores_sha256": "integrated-scores",
                "initial_lora_sha256": "same-lora",
            })
            ig.base.write_json(path / "status.json", {"status": "completed"})
            ig.base.write_json(path / "schedule.json", {
                "source": "F0" if variant in ig.F0_ONLY else "F0+F1",
            })
            six = {
                key: {"n": count, "mean_correct": 0.5, "mean_lp": -1.0}
                for key, count in ig.base.EXPECTED_SET_COUNTS.items()
            }
            summary = {
                "n": 5, "mcqa_accuracy": 1.0,
                "mean_answer_logprob": -2.0,
                "mean_answer_logprob_delta": -0.5,
                "min_answer_logprob_delta": -1.0,
                "mean_mcqa_margin": 0.5,
            }
            metrics = {
                "six_set": {"sets": six},
                "five_set": {"same_object": 0.5, "hop2": 0.5},
                "logprob_audits": {
                    "F1": {"summary": summary},
                    "retain_heldout": {"summary": {"n": 33}},
                },
                "diagnostics": {
                    "updates": 114, "unique_anchor_count": 33,
                    "anchor_schedule_sha256": "same-anchor-plan",
                    "cumulative_retain": {"worst_final_delta": -1.0},
                    "historical_damage_prior": {"prior": {str(k): 1.0 for k in range(28)}},
                },
            }
            ig.base.write_json(path / "metrics.json", metrics)
            paths.append(str(path))
        return paths

    def test_six_variant_comparison_is_read_only_and_complete(self):
        with tempfile.TemporaryDirectory(prefix="gsu-v7-compare-") as folder:
            paths = self.make_runs(folder)
            with unittest.mock.patch.object(ig.base, "train_model") as trainer:
                result = ig.compare_runs(paths, Path(folder) / "report")
            trainer.assert_not_called()
            self.assertTrue(result["complete_screen"])
            self.assertEqual(len(result["rows"]), 6)
            self.assertTrue((Path(folder) / "report" / "comparison.md").is_file())

    def test_compare_requires_fresh_baselines(self):
        with tempfile.TemporaryDirectory(prefix="gsu-v7-compare-") as folder:
            paths = self.make_runs(folder, ("B1", "B2"))
            with self.assertRaisesRegex(RuntimeError, "B0 and B1"):
                ig.compare_runs(paths, Path(folder) / "report")

    def test_compare_rejects_mixed_integrated_scores(self):
        with tempfile.TemporaryDirectory(prefix="gsu-v7-compare-") as folder:
            paths = self.make_runs(folder, ("B0", "B1"))
            ig.base.write_json(Path(paths[1]) / "resolved_config.json", {
                "base_scores_sha256": "not-same",
                "initial_lora_sha256": "same-lora",
            }, overwrite=True)
            with self.assertRaisesRegex(RuntimeError, "frozen code/data/init/score/schedule"):
                ig.compare_runs(paths, Path(folder) / "report")


if __name__ == "__main__":
    unittest.main()

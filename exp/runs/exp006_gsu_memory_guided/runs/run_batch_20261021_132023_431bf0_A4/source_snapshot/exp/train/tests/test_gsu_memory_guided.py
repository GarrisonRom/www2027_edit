"""Offline implementation tests; these are not FaithUn experiment results."""
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from peft import LoraConfig, get_peft_model
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_gsu_memory_guided as mg
import diagnose_gsu_actual_feedback as fb


def examples(count, source="F0"):
    return [{"id": f"{source}:{i}", "source": source, "question": f"q{i}",
             "answer": "a", "false_options": ["b", "c"], "path_count": 1}
            for i in range(count)]


def scores(values=(0.0, 0.0, 0.5, 1.0)):
    return [{"component": f"component:{i}", "shared_score_S": value}
            for i, value in enumerate(values)]


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        values = ([1] if add_special_tokens else []) + [2 + ord(char) % 60 for char in text]
        if return_tensors is None:
            return {"input_ids": values}
        ids = torch.tensor([values], dtype=torch.long)
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


def tiny_base(layers=1):
    config = Qwen2Config(
        vocab_size=64, hidden_size=8, intermediate_size=16,
        num_hidden_layers=layers, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128,
        attention_dropout=0.0, use_cache=False,
    )
    return Qwen2ForCausalLM(config).eval()


def tiny_peft(layers=1):
    base = tiny_base(layers)
    for parameter in base.parameters():
        parameter.requires_grad = False
    return get_peft_model(base, LoraConfig(
        r=mg.LORA_R, lora_alpha=mg.LORA_ALPHA, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=mg.TARGET_MODULES,
    ))


class ScheduleTests(unittest.TestCase):
    def test_equal_114_update_budget(self):
        f0 = examples(33)
        f1 = examples(5, "F1")
        a0 = mg.make_update_schedule(f0, 114, 42)
        a1 = mg.make_update_schedule(f0 + f1, 114, 42)
        self.assertEqual((len(a0), len(a1)), (114, 114))
        a0_counts = mg.schedule_manifest(a0, f0, "F0")["item_counts"]
        a1_counts = mg.schedule_manifest(a1, f0 + f1, "F0+F1")["item_counts"]
        self.assertEqual(sorted(a0_counts.values()), [3] * 18 + [4] * 15)
        self.assertEqual(set(a1_counts.values()), {3})
        self.assertTrue(all(row["source"] == "F0" for row in a0))

    def test_order_matches_legacy_in_place_shuffle(self):
        items = examples(38)
        order = list(items)
        expected = []
        for epoch in range(3):
            random.Random(42 + epoch).shuffle(order)
            expected.extend(item["id"] for item in order)
        actual = mg.make_update_schedule(items, 114, 42)
        self.assertEqual([row["item_id"] for row in actual], expected)

    def test_anchors_identical_to_legacy_rng(self):
        rng = random.Random(42)
        expected = [rng.sample(range(33), 16) for _ in range(114)]
        self.assertEqual(mg.make_anchor_schedule(), expected)
        self.assertEqual(set().union(*map(set, expected)), set(range(33)))

    def test_plan_is_reproducible_and_does_not_mutate_examples(self):
        items = examples(33)
        original = copy.deepcopy(items)
        self.assertEqual(mg.make_update_schedule(items, 114, 42),
                         mg.make_update_schedule(items, 114, 42))
        self.assertEqual(items, original)

    def test_empty_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            mg.make_update_schedule([], 114, 42)


class GuidanceTests(unittest.TestCase):
    def test_all_seven_variants_dispatch(self):
        for variant in mg.VARIANTS:
            guide = mg.build_component_guide(scores(), variant)
            self.assertEqual(guide["applied"], variant not in ("A0", "A1"))
        a2 = mg.build_component_guide(scores(), "A2")
        a4 = mg.build_component_guide(scores(), "A4")
        self.assertEqual(a2["weights"], a4["weights"])

    def test_even_median_and_ties(self):
        guide = mg.build_component_guide(scores(), "A3")
        self.assertEqual(guide["structured_median_S"], 0.25)
        self.assertEqual(guide["structured_hard_kept_components"], 2)
        tied = mg.build_component_guide(scores((0.0, 0.5, 0.5, 1.0)), "A3")
        self.assertEqual(tied["structured_hard_kept_components"], 3)

    def test_random_controls_preserve_distribution_and_cardinality(self):
        structured = mg.build_component_guide(scores(), "A2")
        random_soft = mg.build_component_guide(scores(), "A2R")
        self.assertEqual(sorted(structured["weights"].values()),
                         sorted(random_soft["weights"].values()))
        structured = mg.build_component_guide(scores(), "A3")
        random_hard = mg.build_component_guide(scores(), "A3R")
        self.assertEqual(sum(structured["weights"].values()),
                         sum(random_hard["weights"].values()))

    def test_nonfinite_and_duplicate_scores_rejected(self):
        with self.assertRaises(RuntimeError):
            mg.build_component_guide(scores((float("nan"), 1.0)), "A2")
        with self.assertRaises(ValueError):
            mg.build_component_guide([scores()[0], scores()[0]], "A2")

    def test_all_zero_scores_are_identity(self):
        guide = mg.build_component_guide(scores((0.0,) * 4), "A2")
        self.assertTrue(guide["zero_score_fallback"]["structured_soft"])
        self.assertEqual(set(guide["weights"].values()), {1.0})

    def test_soft_norm_and_hard_gradient_mask(self):
        p, q = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
        raw = {p: torch.tensor([3.0, 4.0]), q: torch.tensor([0.0, 4.0])}
        mapping = {p: "p", q: "q"}
        guided, info = mg.apply_component_guidance([p, q], raw, mapping, {"p": 1.0, "q": 0.5}, True)
        self.assertAlmostEqual(info["raw_norm"], info["guided_norm"], places=5)
        self.assertTrue(torch.allclose(guided[p], raw[p] * info["renorm_scale"]))
        hard, info = mg.apply_component_guidance([p, q], raw, mapping, {"p": 1.0, "q": 0.0}, True)
        self.assertEqual(float(hard[q].abs().sum()), 0.0)
        self.assertAlmostEqual(info["raw_norm"], info["guided_norm"], places=5)

    def test_no_guidance_is_exact_identity(self):
        p = torch.nn.Parameter(torch.zeros(3))
        raw = {p: torch.tensor([0.123, 1.234, 2.345])}
        for apply in (False, True):
            guided, _ = mg.apply_component_guidance([p], raw, {p: "p"}, {"p": 1.0}, apply)
            self.assertTrue(torch.equal(guided[p], raw[p]))

    def test_annihilated_nonzero_gradient_fails_without_silent_fallback(self):
        p = torch.nn.Parameter(torch.zeros(1))
        with self.assertRaises(RuntimeError):
            mg.apply_component_guidance([p], {p: torch.ones(1)}, {p: "p"}, {"p": 0.0}, True)

    def test_known_score_formula(self):
        p, q = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
        groups = {"positive": [p], "negative": [q]}
        f0_grad = {p: torch.tensor([1.0, 0.0]), q: torch.tensor([1.0, 0.0])}
        f1_grad = {p: torch.tensor([1.0, 0.0]), q: torch.tensor([-1.0, 0.0])}
        snapshots = [(f0_grad, -1.0), (f0_grad, -1.0), (f1_grad, -1.0)]
        with mock.patch.object(mg, "cpu_gradient_snapshot", side_effect=snapshots):
            rows, _ = mg.compute_component_scores(None, None, examples(2), examples(1, "F1"), [p, q], groups)
        indexed = {row["component"]: row for row in rows}
        self.assertEqual(indexed["positive"]["shared_score_S"], 1.0)
        self.assertEqual(indexed["negative"]["shared_score_S"], 0.0)
        self.assertEqual(indexed["negative"]["f0_memory_path_consistency_P"], -1.0)


class FeedbackAndAuditTests(unittest.TestCase):
    def execute_linear(self, beneficial=False):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        params = [model.weight]
        optimizer = torch.optim.AdamW(params, lr=0.1)

        def logprob(model, tok, question, answer):
            value = model.weight.sum()
            return -value if beneficial and question == "retain" else value

        with mock.patch.object(mg, "lp_masked", logprob), mock.patch.object(fb, "lp_masked", logprob):
            row = mg.forget_update(
                model, None, params, optimizer, {"question": "forget", "answer": "a"},
                [("retain", "a")], {"applied": False, "weights": {"p": 1.0}}, {model.weight: "p"},
            )
        return row, optimizer.state[model.weight]

    def test_actual_feedback_captures_true_raw_state(self):
        row, state = self.execute_linear()
        self.assertLess(row["anchor_lp_after_raw_adamw"][0], row["anchor_lp_after_feedback"][0])
        self.assertAlmostEqual(row["anchor_lp_delta_raw"][0], -0.1, places=5)
        self.assertLessEqual(row["correction_to_raw_norm_ratio"], 0.300001)
        self.assertLessEqual(row["feedback_round_count"], 2)
        self.assertAlmostEqual(float(state["exp_avg"]), 0.1, places=6)

    def test_no_risk_feedback_is_noop(self):
        row, _ = self.execute_linear(beneficial=True)
        self.assertEqual(row["feedback_round_count"], 0)
        self.assertEqual(row["anchor_lp_after_raw_adamw"], row["anchor_lp_after_feedback"])
        self.assertEqual(row["correction_to_raw_norm_ratio"], 0.0)

    def test_mcqa_and_answer_only_logprob_are_not_mixed(self):
        model = torch.nn.Linear(1, 1)
        with mock.patch.object(mg.te, "score_choices", return_value=[-1.0, -2.0, -3.0]), \
                mock.patch.object(mg, "lp_masked", return_value=torch.tensor(-0.25)):
            row = mg.collect_item_audit(model, None, examples(1))[0]
        self.assertEqual(row["answer_logprob"], -0.25)
        self.assertEqual(row["correct_answer_logprob"], -1.0)
        self.assertEqual(row["mcqa_margin"], 1.0)
        self.assertTrue(model.training)

    def test_cumulative_loss_is_relative_to_initial_not_local_sum(self):
        items = examples(2, "retain")
        initial = {item["id"]: -1.0 for item in items}
        with mock.patch.object(mg, "answer_logprob", side_effect=[-5.0, -2.0]):
            snapshot = mg.cumulative_snapshot(None, None, items, initial, 38)
        self.assertEqual(snapshot["minimum_delta_from_initial"], -4.0)


class ArchiveAndDataTests(unittest.TestCase):
    def test_run_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory(prefix="gsu-test-") as folder:
            args = SimpleNamespace(run_dir=folder, variant="A0", model=None)
            sentinel = Path(folder) / "sentinel.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                mg.run_logged(args)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_failed_run_keeps_status_and_logs(self):
        with tempfile.TemporaryDirectory(prefix="gsu-test-") as folder:
            path = Path(folder) / "new-run"
            args = SimpleNamespace(run_dir=str(path), variant="A0", model="missing")
            with mock.patch.object(mg, "capture_source", return_value={}), \
                    mock.patch.object(mg, "git_output", return_value="test"), \
                    mock.patch.object(mg, "preflight", side_effect=RuntimeError("injected failure")), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    mg.run_logged(args)
            self.assertEqual(mg.read_json(path / "status.json")["status"], "failed")
            self.assertIn("injected failure", (path / "stderr.log").read_text(encoding="utf-8"))

    def test_frozen_F1_missing_path_fails(self):
        fact = {"subject": "s", "relation": "capital", "object": "a", "S": [[1, 0]]}
        candidate = {**fact, "distractors": ["b", "c"]}
        with self.assertRaises(RuntimeError):
            mg.build_f1_items([fact], [candidate], [[2, 0]])

    def test_frozen_F1_distractors_are_not_fabricated(self):
        fact = {"subject": "s", "relation": "capital", "object": "a", "S": [[1, 0]]}
        with self.assertRaises(RuntimeError):
            mg.build_f1_items([fact], [fact], [[1, 0]])


class ComparisonTests(unittest.TestCase):
    def make_runs(self, folder, variants=mg.VARIANTS):
        paths = []
        for variant in variants:
            path = Path(folder) / variant
            path.mkdir()
            config = {key: "same" for key in (
                "model", "seed", "lr", "updates", "lora", "optimizer", "feedback",
                "input_sha256", "source_sha256", "model_metadata_sha256", "model_files",
            )}
            config["variant"] = variant
            mg.write_json(path / "config.json", config)
            mg.write_json(path / "resolved_config.json", {
                "base_scores_sha256": "score", "initial_lora_sha256": "lora",
            })
            mg.write_json(path / "status.json", {"status": "completed"})
            mg.write_json(path / "schedule.json", {"source": "F0" if variant in mg.F0_ONLY else "F0+F1"})
            mg.write_json(path / "metrics.json", {
                "six_set": {"sets": {
                    name: {"n": n, "mean_correct": 0.0, "mean_lp": -1.0}
                    for name, n in mg.EXPECTED_SET_COUNTS.items()
                }},
                "five_set": {"same_object": 0.0, "hop2": 0.0},
                "logprob_audits": {
                    "F1": {"summary": {
                        "mcqa_accuracy": 0.0, "mean_answer_logprob": -1.0,
                        "mean_answer_logprob_delta": -0.5, "min_answer_logprob_delta": -1.0,
                        "mean_mcqa_margin": -1.0,
                    }},
                    "retain_heldout": {"summary": {"n": 33}},
                },
                "diagnostics": {"updates": 114, "unique_anchor_count": 33,
                                "anchor_schedule_sha256": "anchor",
                                "cumulative_retain": {"worst_final_delta": -1.0}},
            })
            paths.append(str(path))
        return paths

    def test_all_seven_runs_are_compared_without_training(self):
        with tempfile.TemporaryDirectory(prefix="gsu-compare-test-") as folder:
            paths = self.make_runs(folder)
            with mock.patch.object(mg, "train_model") as trainer, redirect_stdout(io.StringIO()):
                result = mg.compare_runs(paths, Path(folder) / "report")
            trainer.assert_not_called()
            self.assertTrue(result["complete_screen"])
            self.assertEqual(len(result["rows"]), 7)
            self.assertTrue((Path(folder) / "report" / "comparison.md").is_file())

    def test_incomplete_screen_is_explicit(self):
        with tempfile.TemporaryDirectory(prefix="gsu-compare-test-") as folder:
            paths = self.make_runs(folder, ("A0", "A1"))
            with redirect_stdout(io.StringIO()):
                result = mg.compare_runs(paths, Path(folder) / "report")
            self.assertFalse(result["complete_screen"])
            self.assertIn("A2R", result["missing_variants"])

    def test_missing_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="gsu-compare-test-") as folder:
            paths = self.make_runs(folder, ("A1", "A2"))
            with self.assertRaisesRegex(RuntimeError, "A0 and A1"):
                mg.compare_runs(paths, Path(folder) / "report")

    def test_mixed_score_snapshots_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="gsu-compare-test-") as folder:
            paths = self.make_runs(folder, ("A0", "A1"))
            mg.write_json(Path(paths[1]) / "resolved_config.json", {
                "base_scores_sha256": "different", "initial_lora_sha256": "lora",
            }, overwrite=True)
            with self.assertRaisesRegex(RuntimeError, "frozen inputs"):
                mg.compare_runs(paths, Path(folder) / "report")

    def test_duplicate_variant_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="gsu-compare-test-") as folder:
            paths = self.make_runs(folder, ("A0", "A1"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                mg.compare_runs(paths + paths[:1], Path(folder) / "report")

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="gsu-compare-test-") as folder:
            paths = self.make_runs(folder, ("A0", "A1"))
            output = Path(folder) / "report"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                mg.compare_runs(paths, output)


class TinyModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_component_grouping_and_scoring_do_not_update_parameters(self):
        torch.manual_seed(42)
        model = tiny_peft(layers=2)
        params, groups, _, _ = mg.collect_component_groups(model)
        self.assertEqual((len(groups), len(params)), (14, 28))
        before = mg.parameter_digest(model)
        rows, meta = mg.compute_component_scores(
            model, CharacterTokenizer(), examples(2), examples(1, "F1"), params, groups
        )
        self.assertEqual(before, mg.parameter_digest(model))
        self.assertEqual((len(rows), meta["f0_count"], meta["memory_count"]), (14, 2, 1))

    def test_tiny_end_to_end_all_seven_variants_write_all_audits(self):
        f0, f1, retain = examples(33), examples(5, "F1"), examples(66, "retain")
        sets = {name: [f"{name}:{i}" for i in range(count)]
                for name, count in mg.EXPECTED_SET_COUNTS.items()}
        qa = {key: {"q": "q", "a": "a", "falses": ["b", "c"]}
              for keys in sets.values() for key in keys}
        inputs = {"clusters": [], "indices": {"forget": list(range(33)), "retain": list(range(66))},
                  "F0": f0, "F1": f1, "retain_all": retain, "sets": sets, "qa": qa}
        anchor_plan = [list(range(16)), list(range(16, 32)), [32] + list(range(15))]
        fake_six = {name: {"n": count, "mean_lp": -1.0, "mean_correct": 0.0}
                    for name, count in mg.EXPECTED_SET_COUNTS.items()}
        with tempfile.TemporaryDirectory(prefix="gsu-tiny-test-") as folder:
            initial_hashes, score_hashes = set(), set()
            for variant in mg.VARIANTS:
                with self.subTest(variant=variant):
                    torch.manual_seed(42)
                    path = Path(folder) / variant
                    path.mkdir()
                    args = SimpleNamespace(run_dir=str(path), variant=variant)
                    with mock.patch.object(mg, "TARGET_UPDATES", 3), \
                            mock.patch.object(mg.AutoTokenizer, "from_pretrained", return_value=CharacterTokenizer()), \
                            mock.patch.object(mg.AutoModelForCausalLM, "from_pretrained", return_value=tiny_base()), \
                            mock.patch.object(mg, "make_anchor_schedule", return_value=anchor_plan), \
                            mock.patch.object(mg.te, "evaluate", return_value={"retain": 0.0, "hop2": 0.0, "same_object": 0.0}), \
                            mock.patch.object(mg.ca, "eval_sets", return_value=fake_six), \
                            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        mg.train_model(args, inputs, {"model": "tiny-test-only", "component_count": 7})
                    metrics = mg.read_json(path / "metrics.json")
                    initial_hashes.add(metrics["diagnostics"]["initial_lora_sha256"])
                    score_hashes.add(metrics["diagnostics"]["base_scores_sha256"])
                    self.assertEqual(metrics["diagnostics"]["updates"], 3)
                    self.assertEqual(metrics["diagnostics"]["unique_anchor_count"], 33)
                    self.assertEqual(metrics["logprob_audits"]["retain_all"]["summary"]["n"], 66)
                    self.assertEqual(metrics["logprob_audits"]["retain_heldout"]["summary"]["n"], 33)
                    rows = [json.loads(line) for line in (path / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
                    if variant in mg.F0_ONLY:
                        self.assertTrue(all(row["forget_source"] == "F0" for row in rows))
                    self.assertTrue(all(row["anchor_count"] == 16 for row in rows))
                    config = mg.read_json(path / "resolved_config.json")
                    self.assertEqual(config["training"]["forget_examples"], 33 if variant in mg.F0_ONLY else 38)
                    self.assertEqual(len(mg.read_json(path / "component_scores.json")["components"]), 7)
                    self.assertTrue((path / "adapter" / "adapter_config.json").exists())
                    self.assertTrue((path / "cumulative_audit.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(len(initial_hashes), 1)
            self.assertEqual(len(score_hashes), 1)


if __name__ == "__main__":
    unittest.main()

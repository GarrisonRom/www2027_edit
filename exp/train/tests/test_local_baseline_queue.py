from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce_all_baselines as runner
import run_local_baseline_queue as local


def test_local_manifest_has_no_cloud_dependency(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    result = runner.model_manifest(tmp_path)
    assert result['files']['config.json']['bytes'] == 2
    assert len(result['files']['config.json']['sha256']) == 64


def test_smoke_does_not_claim_capability_or_performance():
    result = local.evaluate_behavior(None, None, None, None, True)
    assert result['performance_evaluation'] == 'not_run'
    assert result['mmlu'] is None and result['ppl'] is None


def test_method_tracks_are_not_silently_lora_klue():
    assert len(local.METHODS) == len(set(local.METHODS)) == 13
    assert runner.rl.profile('KLUE', 'native')['neuron_fraction'] == .05


def test_native_evaluation_discloses_retain_overlap(monkeypatch, tmp_path):
    from types import SimpleNamespace
    behavior = {'groups': {key: {'summary': {'mcqa_accuracy': .25}}
                          for key in ('F0', 'F1', 'retain_heldout')}}
    monkeypatch.setattr(runner.base, 'evaluate', lambda *args: behavior)
    monkeypatch.setattr(runner.routing, 'full_hop2_audit', lambda *args:
                        {'all_hop2': {'summary': {'path_macro_accuracy': .5}}})
    result = local.evaluate_behavior(SimpleNamespace(eval=lambda: None), None, None,
                                     tmp_path, False, retain_is_heldout=False)
    assert result['retain_is_heldout'] is False
    assert result['all89'] == .5
    assert result['capability_status'] == 'not_run_in_local_behavior_protocol'
    assert (tmp_path / 'behavior.json').is_file()

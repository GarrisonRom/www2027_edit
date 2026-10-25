import importlib.util
from pathlib import Path
import sys
import pytest

script=Path(__file__).resolve().parents[1]/'evaluate_cloud_capabilities.py'
sys.path.insert(0,str(script.parent))
spec=importlib.util.spec_from_file_location('cloud_capability_eval',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_sliding_windows_score_each_next_token_once():
    windows=module.token_windows(6000,max_length=2048,stride=512)
    assert sum(window['target_count'] for window in windows)==5999
    scored=[]
    for window in windows:scored.extend(range(window['target_start'],window['end']))
    assert scored==list(range(1,6000))


def test_harness_dtype_metadata_is_json_serializable_without_changing_scores():
    import json
    import torch
    result=module.json_metadata({'config':{'dtype':torch.bfloat16},'accuracy':.625})
    assert json.loads(json.dumps(result))=={'config':{'dtype':'torch.bfloat16'},'accuracy':.625}


def test_window_arguments_are_validated():
    with pytest.raises(ValueError):module.token_windows(10,max_length=8,stride=8)
    with pytest.raises(ValueError):module.token_windows(1)


def test_mmlu_sample_plan_is_subject_balanced_and_reproducible():
    lengths={'history':100,'math':80,'biology':60}
    plan=module.mmlu_sample_plan(lengths,10)
    assert plan==module.mmlu_sample_plan(lengths,10)
    assert all(len(ids)==10 and len(set(ids))==10 for ids in plan.values())
    assert all(max(ids)<lengths[name] for name,ids in plan.items())

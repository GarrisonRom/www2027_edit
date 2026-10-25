import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_cloud_queue_uses_local_worker():
    source = (ROOT / "run_cloud_c1_reproduction.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'str(c.WS/"exp/train/run_local_gsu_c_group.py")' in source
    assert '"--cloud_architecture","--variant","C1"' in source
    assert "batch_20261023_03_local_aligned" in source

def test_shared_worker_keeps_local_training_contract():
    source = (ROOT / "run_local_gsu_c_group.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
    body = ast.get_source_segment(source, worker)
    assert 'task_type="CAUSAL_LM"' in body
    assert "model.train()" in body
    assert "update in (38, 76, 114)" in body
    assert 'updates = 8 if args.smoke else CONTRACT["updates"]' in body
    assert "row = routed.train_step" in body
    assert 'args.variant in ("C2", "C3", "C3R", "C3D", "C4", "C2D")' in body

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continue_local_baselines as continuation


def test_hash_matches_standard_sha256(tmp_path):
    import hashlib
    path = tmp_path / 'input'
    path.write_bytes(b'faithun')
    assert continuation.sha(path) == hashlib.sha256(b'faithun').hexdigest()


def test_only_missing_methods_are_scheduled():
    assert len(continuation.METHODS) == 7
    assert not set(continuation.METHODS) & {'GA','KLUE','GradDiff','NPO','SimNPO','RMU'}


def test_comparison_does_not_include_incomplete_formal(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, 'collect_rows', lambda roots: [
        dict(method='KLUE',all89=.5,covered75=3,f0=1,f1=2,retain=28,
             heldout=False,source='raw/metrics.json')])
    continuation.summarize([],tmp_path)
    text = (tmp_path/'comparison.md').read_text(encoding='utf-8')
    assert 'trained' in text and 'False' in text and 'raw/metrics.json' in text
    assert json.loads((tmp_path/'comparison.json').read_text(encoding='utf-8'))[0]['heldout'] is False

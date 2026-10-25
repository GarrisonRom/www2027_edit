import importlib.util
import io
from pathlib import Path
import tarfile
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = load('cloud_comparison', 'exp/analysis/compare_cloud_baselines.py')
sync = load('workspace_sync', 'tools/sync_research_workspace.py')


def test_direction_and_competition_ties():
    assert comparison.ranks({'a':.1,'b':.2,'c':.2,'d':.4},'min') == {'a':1,'b':2,'c':2,'d':4}
    assert comparison.ranks({'a':.1,'b':.4},'max') == {'a':2,'b':1}


def test_missing_and_audit_not_ranked():
    assert comparison.ranks({'unrun_GSU':None,'measured':.4},'min') == {'measured':1}
    assert comparison.ranks({'support_fact':.4},'audit') == {}


def test_archive_traversal_is_rejected(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        entry=tarfile.TarInfo('../escape.txt');entry.size=1
        archive.addfile(entry,io.BytesIO(b'x'))
    with pytest.raises(RuntimeError,match='Unsafe archive'):
        sync.safe_unpack(buffer.getvalue(),tmp_path/'new')
    assert not (tmp_path/'escape.txt').exists()


def test_archive_never_overwrites(tmp_path):
    with pytest.raises(FileExistsError):
        sync.safe_unpack(b'',tmp_path)

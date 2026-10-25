"""Load only hash-pinned historical primitives; new protocol lives in baselines/."""
import sys
from .bootstrap import ROOT,verify


def load():
    verify()
    path=ROOT/'vendor/runtime/exp/train'
    sys.path.insert(0,str(path))
    import reproduce_all_baselines as core
    if str(path) not in str(core.__file__):
        raise RuntimeError('Another baseline implementation is already imported; use a fresh process')
    return core

"""Audited baseline training and evaluation, independent of historical pilot runs."""
from pathlib import Path
import sys
deps=Path(__file__).resolve().parent/'.deps'
if deps.exists():sys.path.insert(0,str(deps))

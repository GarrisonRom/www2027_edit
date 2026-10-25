import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'pareto_audit', Path(__file__).resolve().parents[2] / 'analysis/audit_gsu_pareto.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_tradeoffs_are_not_dominance():
    objectives = {'recall': 'min', 'utility': 'max'}
    a = {'recall': .2, 'utility': .4}
    b = {'recall': .3, 'utility': .6}
    assert not audit.dominates(a, b, objectives)
    assert not audit.dominates(b, a, objectives)
    assert not audit.dominates(a, a, objectives)


def test_equal_utility_and_lower_recall_dominate():
    objectives = {'recall': 'min', 'utility': 'max'}
    a = {'method': 'expanded', 'recall': 22 / 75, 'utility': 14 / 33}
    b = {'method': 'B2', 'recall': 44 / 75, 'utility': 14 / 33}
    result = audit.mark_frontier([a, b], objectives)
    assert result[1]['dominated_by'] == ['expanded']
    assert result[0]['empirically_nondominated']

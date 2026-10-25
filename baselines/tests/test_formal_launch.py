import json
from baselines.launch_formal import METHODS,write,summarize
from baselines.recipes import lora_adapt,recipe,validate


def test_formal_methods_exclude_unready_or_unvalidated_methods():
    assert len(METHODS)==11 and len(set(METHODS))==11
    assert 'PDU' not in METHODS and 'KLUE' not in METHODS
    for m in METHODS:validate(lora_adapt(recipe(m)))


def test_summary_does_not_claim_training_is_complete_evaluation(tmp_path):
    folder=tmp_path/'GA_formal';folder.mkdir()
    write(folder/'training_summary.json',dict(updates=10,final_training_ua=.2,stop='threshold'))
    state=dict(jobs=[dict(method='GA',phase='training',status='completed',artifact=str(folder))])
    summarize(tmp_path,state)
    data=json.loads((tmp_path/'comparison.json').read_text())
    assert data['rows'][0]['phase']=='training'
    assert 'complete' not in data['rows'][0]

import pytest
from baselines.download_modelscope_eval import safe_name


@pytest.mark.parametrize('name',['/tmp/file','../file','data/../../file','C:/file','data'+chr(92)+'file'])
def test_archive_path_traversal_rejected(name):
    with pytest.raises(ValueError):safe_name(name)


def test_standard_archive_path_accepted():
    assert str(safe_name('data/test/abstract_algebra_test.csv'))=='data/test/abstract_algebra_test.csv'


def test_local_task_override_keeps_prompt_and_split_contract(monkeypatch,tmp_path):
    import baselines.local_capability_data as data
    from copy import deepcopy
    source=dict(task='mmlu_abstract_algebra',dataset_path='cais/mmlu',dataset_name='abstract_algebra',
        test_split='test',fewshot_split='dev',fewshot_config={'sampler':'first_n'},
        doc_to_text='original question/choices/Answer template',doc_to_choice=['A','B','C','D'],doc_to_target='answer')
    class Manager:
        def _get_config(self,name):return deepcopy(source)
    monkeypatch.setattr(data,'checked_files',lambda *args:{'dev':'dev.parquet','test':'test.parquet','validation':'val.parquet'})
    config=data.mmlu_task_config(Manager(),'mmlu_abstract_algebra',tmp_path)
    assert config['dataset_path']=='parquet' and config['dataset_name'] is None
    for key in ('task','test_split','fewshot_split','fewshot_config','doc_to_text','doc_to_choice','doc_to_target'):
        assert config[key]==source[key]
    assert source['dataset_path']=='cais/mmlu'
    with pytest.raises(ValueError):data.mmlu_task_config(Manager(),'mmlu_wrong_subject',tmp_path)

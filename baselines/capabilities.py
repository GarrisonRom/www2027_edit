"""Full MMLU counterpart to the cloud sampled-MMLU audit."""
import hashlib
import json


def full_mmlu(model,tokenizer,output_dir):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager,get_task_dict
    task_tree=get_task_dict(['mmlu'],task_manager=TaskManager());datasets={}
    def visit(tree):
        for key,value in tree.items():
            if isinstance(value,dict):visit(value)
            else:
                name=value.dump_config().get('task',str(key))
                datasets[name]={split:getattr(ds,'_fingerprint',None) for split,ds in value.dataset.items()}
    visit(task_tree)
    if len(datasets)!=57:raise RuntimeError('Full MMLU must contain57 subjects')
    harness=HFLM(model,tokenizer=tokenizer,batch_size=1,max_batch_size=1,device='cuda')
    result=simple_evaluate(model=harness,tasks=['mmlu'],num_fewshot=5,batch_size=1,
        log_samples=True,write_out=True,apply_chat_template=False,random_seed=42,
        numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42,bootstrap_iters=0)
    subjects={};path=output_dir/'mmlu_full_items.jsonl'
    with path.open('x',encoding='utf-8') as log:
        for name,samples in result['samples'].items():
            if name=='mmlu':continue
            metrics=result['results'][name]
            accuracies=[float(v) for k,v in metrics.items() if k.split(',')[0]=='acc']
            if len(accuracies)!=1:raise RuntimeError('Missing/ambiguous MMLU accuracy: '+name)
            subjects[name]=dict(accuracy=accuracies[0],n=len(samples))
            for row in samples:log.write(json.dumps(dict(subject=name,sample=row),default=str,ensure_ascii=False)+'\n')
    if len(subjects)!=57:raise RuntimeError('Incomplete full MMLU result')
    total=sum(v['n'] for v in subjects.values())
    return dict(protocol='full MMLU57subjects,5-shot,no chat,seed42',full=True,subjects=subjects,
        dataset_fingerprints=datasets,evaluated_samples=total,
        macro_subject_accuracy=sum(v['accuracy'] for v in subjects.values())/57,
        micro_accuracy=sum(v['accuracy']*v['n'] for v in subjects.values())/total,
        items_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

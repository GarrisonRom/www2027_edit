"""Development task from unused clusters. Never reads official test outcomes."""
import torch
import decisive_controls as controls


def task_inputs(inputs,plan):
    protocol=__import__('run_remote_baselines').protocol
    result=dict(inputs)
    result['F0']=protocol.build_f0_items(inputs['clusters'],plan['forget'])
    result['retain_all']=protocol.build_retain_items(inputs['clusters'],plan['retain'])
    result['F1']=[]
    result['development_plan']=plan
    return result


@torch.no_grad()
def evaluate(model,tok,inputs):
    import run_remote_baselines as b
    plan=inputs['development_plan'];clusters=inputs['clusters']
    previous=model.training;model.eval()
    def correct(q,a,falses):
        scores=b.losses.te.score_choices(model,tok,q,[a]+list(falses))
        return int(scores[0]>max(scores[1:]))
    direct=[correct(x['question'],x['answer'],x['false_options']) for x in inputs['F0']]
    retain=[correct(x['question'],x['answer'],x['false_options']) for x in inputs['retain_all']]
    paths=[]
    for i in plan['forget']:
        fact=clusters[i][2]['hop2_test']
        for j,questions in enumerate(fact['questions']):
            values=[correct(q,fact['answers'][j],fact['false_answer_options'][j]) for q in questions]
            if values:paths.append(sum(values)/len(values))
    if not paths:raise RuntimeError('Development target lacks multihop paths')
    model.train(previous)
    return {'direct_accuracy':sum(direct)/len(direct),'retain_accuracy':sum(retain)/len(retain),
            'multihop_macro':sum(paths)/len(paths),'path_count':len(paths),
            'direct_n':len(direct),'retain_n':len(retain),'scope':'development only, official test outcomes not evaluated'}


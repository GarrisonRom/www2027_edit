"""FaithUn Figure8 prompt, deterministic options and answer-conditional scoring."""
import hashlib
import json
import random
import torch
import torch.nn.functional as F

INSTRUCTION=('Answer the following question by simply selecting a proper answer among the given '
    'options. You must generate only the exact word without an explanation.')


def option_order(item,seed=42):
    values=[item['answer']]+list(item['false_options'])
    if len(values)!=3 or len(set(values))!=3:raise ValueError('Need three distinct options')
    identity=json.dumps([item['question'],item['answer'],sorted(item['false_options'])],ensure_ascii=False)
    fingerprint=hashlib.sha256(identity.encode('utf-8')).hexdigest()
    # Training-UA and evaluation may use different bookkeeping IDs for the same QA.
    random.Random(f"faithun-{seed}-{fingerprint}").shuffle(values)
    return values


def prompt(item,seed=42):
    # Figure8 does not specify a serialized list delimiter. Freeze this convention explicitly.
    options=json.dumps(option_order(item,seed),ensure_ascii=False)
    return f"{INSTRUCTION}\nQuestion: {item['question']}\nOptions: {options}\nYour Answer:"


def encode(tok,item,device,prompted=False):
    q=prompt(item) if prompted else item['question']
    head=tok(q,add_special_tokens=True)['input_ids']
    full=tok(q+' '+item['answer'],add_special_tokens=True)['input_ids']
    if full[:len(head)]!=head:raise ValueError('Tokenizer changes question/answer boundary')
    ids=torch.tensor([full],device=device);labels=ids.clone();labels[:,:len(head)]=-100
    if not bool((labels[:,1:]!=-100).any()):raise ValueError('Empty answer')
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels)


@torch.no_grad()
def scores(model,tok,item,prompted=True):
    sums=[];means=[];legacy=[]
    context=prompt(item) if prompted else item['question']
    for answer in [item['answer']]+list(item['false_options']):
        # All candidates MUST share the same question/options. Do not rebuild options
        # after replacing the correct answer by a false candidate.
        b=encode(tok,dict(item,question=context,answer=answer),model.device,False)
        logits=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits[:,:-1].float()
        labels=b['labels'][:,1:];valid=labels!=-100
        nll=F.cross_entropy(logits.transpose(1,2),labels,ignore_index=-100,reduction='none')
        sums.append(-float(nll.sum()));means.append(-float(nll.sum()/valid.sum()))
        all_nll=F.cross_entropy(logits.transpose(1,2),b['input_ids'][:,1:],reduction='mean')
        legacy.append(-float(all_nll))
    return dict(answer_sum=sums,answer_mean=means,full_sequence_mean=legacy)


def classify(values):
    margin=values[0]-max(values[1:])
    return dict(correct=int(margin>0),margin=margin,tie=margin==0)


def dataset_manifest(items):
    return hashlib.sha256(json.dumps(items,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

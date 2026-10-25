"""Matched random controls and isolated unused-cluster development protocol."""
import hashlib
import math
import random
import torch


def permute_within_type(weights,seed):
    """Only the component-score assignment changes; module-type histogram is exact."""
    groups={}
    for name in sorted(weights):groups.setdefault(name.rsplit('.',1)[-1],[]).append(name)
    result=dict(weights)
    rng=random.Random(seed+7919)
    for names in groups.values():
        values=[weights[name] for name in names];rng.shuffle(values)
        result.update(zip(names,values))
    return result


def random_subspaces(bases,names,seed):
    """Same rank/shape/None per tensor; deterministic name-seeded Gaussian QR."""
    result={};meta={}
    for param,basis in bases.items():
        name=names[param]
        if basis is None:
            result[param]=None;meta[name]={'rank':None,'identity_fallback':True};continue
        rank=basis.shape[1]
        token=int.from_bytes(hashlib.sha256((str(seed)+'|'+name).encode()).digest()[:8],'big')%(2**63)
        generator=torch.Generator(device='cpu').manual_seed(token)
        matrix=torch.randn(param.numel(),rank,generator=generator)
        q,_=torch.linalg.qr(matrix,mode='reduced')
        result[param]=q.to(param.device)
        meta[name]={'rank':rank,'identity_fallback':False,'random_seed':token}
    return result,meta


def unused_development_split(clusters,indices):
    excluded=set().union(*(set(v) for v in indices.values()))
    official_facts={tuple(clusters[i][2]['hop1_unlearn']['graphs']) for i in excluded}
    candidates=[i for i in range(len(clusters)) if i not in excluded
                and tuple(clusters[i][2]['hop1_unlearn']['graphs']) not in official_facts]
    # Different aliases of the same direct triple are not allowed across dev roles.
    unique=[];seen=set()
    for i in sorted(candidates,key=lambda j:hashlib.sha256(('gsu-dev-v1|'+str(j)).encode()).hexdigest()):
        fact=tuple(clusters[i][2]['hop1_unlearn']['graphs'])
        if fact not in seen:seen.add(fact);unique.append(i)
    if len(unique)<48:raise RuntimeError('Need48 unused nonduplicate clusters; do not borrow official test')
    return {'forget':unique[:24],'retain':unique[24:48],
            'unused_rest':unique[48:],'official_excluded':sorted(excluded),
            'limitations':'cluster/direct-triple disjoint, not guaranteed entity/graph-disjoint; development only'}


def select_trial(rows):
    """Fixed gates, then lower multihop recall. No fallback winner if all fail."""
    eligible=[r for r in rows if r.get('status')=='completed' and r['direct_accuracy']<=1/3+1e-12
              and r['retain_accuracy']>=r['original_retain_accuracy']-.10-1e-12]
    if not eligible:return {'status':'no_feasible_configuration','selected':None}
    best=min(eligible,key=lambda r:(r['multihop_macro'],r['direct_accuracy'],r['learning_rate']))
    return {'status':'selected','selected':best,'rule':'UA<=1/3, retain drop<=10pp, min dev multihop; tie direct then smaller LR'}


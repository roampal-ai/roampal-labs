#!/usr/bin/env python
"""Test: do facts add retrieval value on top of summaries?
Uses Wilson+CE archived DB (has both facts and summaries with real scores)."""
import chromadb, json, math, sys, os
import numpy as np
from sentence_transformers import CrossEncoder
from scipy import stats

os.environ['PYTHONUTF8'] = '1'
sys.stdout.reconfigure(encoding='utf-8')

client = chromadb.PersistentClient(path='archive/pre_fix_run/runs/02.Wilson+CE')
col = client.list_collections()[0]
print(f'DB: {col.count()} memories')

data = json.loads(open('data/locomo_full.json', encoding='utf-8').read())
non_adv = [q for q in data['locomo_exam'] if q.get('category_name') != 'adversarial' and q.get('ground_truth','').strip()]
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

def check_relevance(content, gt):
    if not gt.strip(): return False
    stop = {'the','a','an','is','was','are','were','be','been','have','has','had','do','does','did',
            'will','would','to','of','in','for','on','with','at','by','from','as','and','but','or',
            'not','no','so','if','that','this','it','they','she','he','we','you','my','me','i',
            'its','them','their','her','him','his','us','our','your'}
    kws = set(gt.lower().split()) - stop
    if len(kws) < 2: kws = set(gt.lower().split())
    return sum(1 for kw in kws if kw in content.lower()) >= min(3, len(kws))

def get_lane(col, question, ce_model, where, n=20, top_k=4):
    """Get top-k from one lane."""
    try:
        r = col.query(query_texts=[question], n_results=n, where=where,
                     include=['documents','metadatas','distances'])
    except Exception:
        return []
    if not r['ids'][0]: return []
    cands = [{'content': r['documents'][0][i], 'dist': r['distances'][0][i]}
             for i in range(len(r['ids'][0]))]
    pairs = [[question, c['content']] for c in cands]
    ces = ce_model.predict(pairs).tolist()
    for i, c in enumerate(cands): c['ce'] = ces[i]
    cands.sort(key=lambda c: c['ce'], reverse=True)
    return cands[:top_k]

# Test configs: vary summary and fact slot counts
configs = [
    ('4sum_0fact', 4, 0),
    ('4sum_1fact', 4, 1),
    ('4sum_2fact', 4, 2),
    ('4sum_3fact', 4, 3),
    ('4sum_4fact', 4, 4),
    ('3sum_4fact', 3, 4),
    ('2sum_4fact', 2, 4),
    ('0sum_4fact', 0, 4),
]

results = {name: {'hits': 0, 'mrr_sum': 0.0, 'per_slot_hits': [0]*8} for name, _, _ in configs}
total = 0

print(f'Testing {len(non_adv)} Qs, {len(configs)} slot configs...')
for qi, q in enumerate(non_adv):
    question = q.get('question', q.get('query',''))
    gt = q.get('ground_truth','')

    # Get both lanes (always fetch max, then truncate per config)
    summaries = get_lane(col, question, ce, {"type": {"$ne": "fact"}}, top_k=4)
    facts = get_lane(col, question, ce, {"type": "fact"}, top_k=4)

    total += 1
    for name, n_sum, n_fact in configs:
        combined = summaries[:n_sum] + facts[:n_fact]
        rr = 0
        for rank, c in enumerate(combined):
            rel = check_relevance(c['content'], gt)
            if rel and rr == 0:
                rr = 1.0 / (rank + 1)
            if rel and rank < 8:
                results[name]['per_slot_hits'][rank] += 1
        results[name]['mrr_sum'] += rr
        if rr > 0: results[name]['hits'] += 1

    if (qi+1) % 500 == 0:
        print(f'  {qi+1}/{len(non_adv)}...')

print(f'\n{"="*70}')
print(f'FACTS VALUE TEST ({total} Qs, Wilson+CE DB)')
print(f'{"="*70}')
print(f'{"Config":<15} {"Hit":>6} {"MRR":>7}  Per-slot cumulative hits')
print(f'{"-"*70}')
for name, n_sum, n_fact in configs:
    r = results[name]
    hit = r['hits'] / total
    mrr = r['mrr_sum'] / total
    slots = r['per_slot_hits']
    cum = []
    running = 0
    for s in slots[:n_sum+n_fact]:
        running += s
        cum.append(f'{running/total:.1%}')
    print(f'{name:<15} {hit:>5.1%} {mrr:>6.3f}  {" → ".join(cum)}')

# McNemar: 4sum_0fact vs 4sum_4fact
print(f'\nMcNemar: 4sum_0fact vs 4sum_4fact')
a_hits = []
b_hits = []
for qi, q in enumerate(non_adv):
    question = q.get('question', q.get('query',''))
    gt = q.get('ground_truth','')
    summaries = get_lane(col, question, ce, {"type": {"$ne": "fact"}}, top_k=4)
    facts = get_lane(col, question, ce, {"type": "fact"}, top_k=4)

    a = any(check_relevance(c['content'], gt) for c in summaries[:4])
    b = any(check_relevance(c['content'], gt) for c in summaries[:4] + facts[:4])
    a_hits.append(1 if a else 0)
    b_hits.append(1 if b else 0)

    if (qi+1) % 500 == 0:
        print(f'  McNemar pass: {qi+1}/{len(non_adv)}...')

oa = sum(1 for x,y in zip(a_hits, b_hits) if x==1 and y==0)
ob = sum(1 for x,y in zip(a_hits, b_hits) if x==0 and y==1)
disc = oa + ob
mcnemar = {}
if disc > 0:
    chi2 = (abs(oa-ob)-1)**2/disc; p = 1-stats.chi2.cdf(chi2,1)
    print(f'  only_sum={oa}, only_sum+fact={ob}, chi2={chi2:.2f}, p={p:.4f}')
    mcnemar = {"only_sum": oa, "only_sum_fact": ob, "chi2": round(chi2, 4), "p": round(p, 6)}
else:
    print('  IDENTICAL')
    mcnemar = {"only_sum": 0, "only_sum_fact": 0, "chi2": 0, "p": 1.0}

# Save structured results
output = {"test": "facts_value", "n_questions": total,
          "db_path": "archive/pre_fix_run/runs/02.Wilson+CE", "configs": {}, "mcnemar_4sum0fact_vs_4sum4fact": mcnemar}
for name, n_sum, n_fact in configs:
    r = results[name]
    output["configs"][name] = {
        "hit_rate": round(r['hits'] / total, 4),
        "mrr": round(r['mrr_sum'] / total, 4),
        "hits": r['hits'], "total": total,
        "n_summaries": n_sum, "n_facts": n_fact,
    }
import pathlib
out_path = pathlib.Path("results/facts_value_results.json")
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f'\nSaved: {out_path}')

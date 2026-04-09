#!/usr/bin/env python
"""Diagnose WHY 33% of questions fail to retrieve the right answer.
For each missed question: is the answer in the DB at all? If so, where does it rank?"""
import chromadb, json, sys, os
import numpy as np
from sentence_transformers import CrossEncoder
from collections import defaultdict

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

def get_lane(col, question, ce_model, where, top_k=20):
    try:
        r = col.query(query_texts=[question], n_results=top_k, where=where,
                     include=['documents','distances'])
    except: return []
    if not r['ids'][0]: return []
    cands = [{'content': r['documents'][0][i], 'dist': r['distances'][0][i]}
             for i in range(len(r['ids'][0]))]
    pairs = [[question, c['content']] for c in cands]
    ces = ce_model.predict(pairs).tolist()
    for i, c in enumerate(cands): c['ce'] = ces[i]
    cands.sort(key=lambda c: c['ce'], reverse=True)
    return cands

# Failure categories
failures = {
    'not_in_db': [],           # Answer not found anywhere in top 100
    'in_facts_deep': [],       # In fact lane but ranked below slot 2
    'in_summaries_deep': [],   # In summary lane but ranked below slot 4
    'in_both_deep': [],        # In both lanes but ranked too deep
    'hit': [],                 # Actually found (for comparison)
}

category_failures = defaultdict(lambda: defaultdict(int))
total = 0

print(f'Diagnosing {len(non_adv)} questions...')
for qi, q in enumerate(non_adv):
    question = q.get('question', q.get('query',''))
    gt = q.get('ground_truth','')
    cat = q.get('category_name', 'unknown')

    # Get top 4 summaries + top 2 facts (our config)
    sums = get_lane(col, question, ce, {"type": {"$ne": "fact"}}, top_k=4)
    facts = get_lane(col, question, ce, {"type": "fact"}, top_k=2)
    combined = sums + facts

    total += 1

    # Check if hit in top 6
    hit_in_top6 = any(check_relevance(c['content'], gt) for c in combined)

    if hit_in_top6:
        failures['hit'].append(qi)
        category_failures[cat]['hit'] += 1
        continue

    # MISS — diagnose why
    # Check deeper: top 100 from each lane
    deep_sums = get_lane(col, question, ce, {"type": {"$ne": "fact"}}, top_k=100)
    deep_facts = get_lane(col, question, ce, {"type": "fact"}, top_k=100)

    found_in_sums = -1
    for rank, c in enumerate(deep_sums):
        if check_relevance(c['content'], gt):
            found_in_sums = rank
            break

    found_in_facts = -1
    for rank, c in enumerate(deep_facts):
        if check_relevance(c['content'], gt):
            found_in_facts = rank
            break

    if found_in_sums == -1 and found_in_facts == -1:
        failures['not_in_db'].append({
            'q': question[:100], 'gt': gt[:100], 'cat': cat
        })
        category_failures[cat]['not_in_db'] += 1
    elif found_in_facts >= 0 and found_in_sums == -1:
        failures['in_facts_deep'].append({
            'q': question[:100], 'gt': gt[:100], 'cat': cat, 'rank': found_in_facts
        })
        category_failures[cat]['in_facts_deep'] += 1
    elif found_in_sums >= 0 and found_in_facts == -1:
        failures['in_summaries_deep'].append({
            'q': question[:100], 'gt': gt[:100], 'cat': cat, 'rank': found_in_sums
        })
        category_failures[cat]['in_summaries_deep'] += 1
    else:
        failures['in_both_deep'].append({
            'q': question[:100], 'gt': gt[:100], 'cat': cat,
            'sum_rank': found_in_sums, 'fact_rank': found_in_facts
        })
        category_failures[cat]['in_both_deep'] += 1

    if (qi+1) % 500 == 0:
        print(f'  {qi+1}/{len(non_adv)}...')

# Results
hit = len(failures['hit'])
not_in_db = len(failures['not_in_db'])
in_facts = len(failures['in_facts_deep'])
in_sums = len(failures['in_summaries_deep'])
in_both = len(failures['in_both_deep'])
total_miss = not_in_db + in_facts + in_sums + in_both

print(f'\n{"="*60}')
print(f'RETRIEVAL FAILURE DIAGNOSIS ({total} Qs, 4sum+2fact)')
print(f'{"="*60}')
print(f'  HIT (retrieved):     {hit:>5} ({hit/total:.1%})')
print(f'  MISS total:          {total_miss:>5} ({total_miss/total:.1%})')
print(f'    NOT IN DB:         {not_in_db:>5} ({not_in_db/total:.1%}) — fact never stored')
print(f'    In facts, deep:    {in_facts:>5} ({in_facts/total:.1%}) — stored but CE ranked too low')
print(f'    In summaries, deep:{in_sums:>5} ({in_sums/total:.1%}) — in summary but ranked too low')
print(f'    In both, deep:     {in_both:>5} ({in_both/total:.1%}) — in both but ranked too low')

ranking_problem = in_facts + in_sums + in_both
storage_problem = not_in_db
print(f'\n  STORAGE PROBLEM:     {storage_problem:>5} ({storage_problem/total:.1%})')
print(f'  RANKING PROBLEM:     {ranking_problem:>5} ({ranking_problem/total:.1%})')

# Rank distribution for ranking failures
if in_facts > 0:
    ranks = [f['rank'] for f in failures['in_facts_deep']]
    print(f'\n  Fact rank distribution (missed): min={min(ranks)}, max={max(ranks)}, median={sorted(ranks)[len(ranks)//2]}')
    buckets = {'3-5': 0, '6-10': 0, '11-20': 0, '21-50': 0, '50+': 0}
    for r in ranks:
        if r < 5: buckets['3-5'] += 1
        elif r < 10: buckets['6-10'] += 1
        elif r < 20: buckets['11-20'] += 1
        elif r < 50: buckets['21-50'] += 1
        else: buckets['50+'] += 1
    print(f'  Rank buckets: {dict(buckets)}')

# Per-category breakdown
print(f'\nPer-category breakdown:')
print(f'{"Category":<15} {"Hit":>5} {"NotInDB":>8} {"Deep":>5} {"Total":>6}')
for cat in sorted(category_failures.keys()):
    cf = category_failures[cat]
    h = cf.get('hit', 0)
    nid = cf.get('not_in_db', 0)
    deep = cf.get('in_facts_deep', 0) + cf.get('in_summaries_deep', 0) + cf.get('in_both_deep', 0)
    t = h + nid + deep
    print(f'{cat:<15} {h:>5} {nid:>8} {deep:>5} {t:>6}')

# Show some examples of NOT IN DB
print(f'\nSample NOT IN DB failures:')
for f in failures['not_in_db'][:5]:
    print(f'  [{f["cat"]}] Q: {f["q"]}')
    print(f'           GT: {f["gt"]}')
    print()

# Save structured results
import pathlib
rank_dist = {}
if in_facts > 0:
    ranks = [f['rank'] for f in failures['in_facts_deep']]
    rank_dist = {'3-5': sum(1 for r in ranks if r < 5),
                 '6-10': sum(1 for r in ranks if 5 <= r < 10),
                 '11-20': sum(1 for r in ranks if 10 <= r < 20),
                 '21-50': sum(1 for r in ranks if 20 <= r < 50),
                 '50+': sum(1 for r in ranks if r >= 50)}

output = {
    "test": "retrieval_failure_diagnosis",
    "n_questions": total,
    "db_path": "archive/pre_fix_run/runs/02.Wilson+CE",
    "config": "4sum_2fact",
    "summary": {
        "hit": hit, "hit_rate": round(hit/total, 4),
        "miss": total_miss, "miss_rate": round(total_miss/total, 4),
        "not_retrievable": not_in_db, "not_retrievable_rate": round(not_in_db/total, 4),
        "ranking_failure": ranking_problem, "ranking_rate": round(ranking_problem/total, 4),
    },
    "ranking_detail": {
        "in_facts_deep": in_facts,
        "in_summaries_deep": in_sums,
        "in_both_deep": in_both,
        "fact_rank_distribution": rank_dist,
    },
    "per_category": {cat: dict(cf) for cat, cf in category_failures.items()},
}
out_path = pathlib.Path("results/retrieval_failure_diagnosis_results.json")
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f'\nSaved: {out_path}')

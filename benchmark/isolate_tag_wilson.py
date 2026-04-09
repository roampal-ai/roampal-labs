#!/usr/bin/env python
"""Isolate what's actually helping: tags, Wilson sort, or both?
4 configs on the same ER DB, two-lane retrieval."""
import asyncio, sys, os
import numpy as np
from scipy import stats

os.environ['PYTHONUTF8'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, chromadb, math, re
from sentence_transformers import CrossEncoder
from typing import List, Set


def check_relevance(content, gt):
    if not gt.strip(): return False
    stop = {'the','a','an','is','was','are','were','be','been','have','has','had','do','does','did',
            'will','would','to','of','in','for','on','with','at','by','from','as','and','but','or',
            'not','no','so','if','that','this','it','they','she','he','we','you','my','me','i',
            'its','them','their','her','him','his','us','our','your'}
    kws = set(gt.lower().split()) - stop
    if len(kws) < 2: kws = set(gt.lower().split())
    return sum(1 for kw in kws if kw in content.lower()) >= min(3, len(kws))


def wilson_lower(s, n, z=1.96):
    if n == 0: return 0.5
    p = max(0, min(1, s)); d = 1 + z**2/n; c = p + z**2/(2*n)
    return (c - z*math.sqrt((p*(1-p)+z**2/(4*n))/n)) / d


def calc_wilson_blend(meta):
    raw = float(meta.get('score', 0.5))
    uses = int(meta.get('uses', 0))
    sc = float(meta.get('success_count', 0.0))
    if uses == 0: return raw
    rate = sc / uses if uses > 0 else 0.5
    w = wilson_lower(rate, uses)
    if uses < 3: return (1-uses/3)*raw + (uses/3)*w
    return w


def match_query_tags(query, known_tags):
    query_lower = query.lower()
    matches = []
    for tag in sorted(known_tags, key=len, reverse=True):
        if len(tag) < 3: continue
        if re.search(r'\b' + re.escape(tag) + r'\b', query_lower):
            matches.append(tag)
    return matches[:8]


def retrieve_twolane(cols, question, ce_model, known_tags,
                     use_tags=False, use_wilson_sort=False, top_k=4):
    """Two-lane retrieval with configurable tags and Wilson sort."""
    all_mems = []

    for lane_type in ['summary', 'fact']:
        where = {"type": "fact"} if lane_type == 'fact' else {"type": {"$ne": "fact"}}

        # Query all tiers
        candidates = []
        for col_name, col in cols.items():
            if col.count() == 0: continue
            try:
                r = col.query(query_texts=[question], n_results=20, where=where,
                             include=['documents', 'metadatas', 'distances'])
            except Exception:
                r = col.query(query_texts=[question], n_results=20,
                             include=['documents', 'metadatas', 'distances'])
            if not r['ids'][0]: continue
            for i in range(len(r['ids'][0])):
                candidates.append({
                    'content': r['documents'][0][i],
                    'meta': r['metadatas'][0][i],
                    'dist': r['distances'][0][i],
                })

        if not candidates: continue

        if use_tags and known_tags:
            query_tags = match_query_tags(question, known_tags)
            if query_tags:
                # Tag match + overlap
                tagged = []
                untagged = []
                for c in candidates:
                    mem_tags = c['meta'].get('tags', '')
                    mem_tag_set = set(t.strip().lower() for t in mem_tags.split('|') if t.strip())
                    overlap = sum(1 for qt in query_tags if qt.lower() in mem_tag_set)
                    if overlap > 0:
                        c['overlap'] = overlap
                        tagged.append(c)
                    else:
                        untagged.append(c)

                if tagged:
                    # Sort tagged by overlap, then tiebreaker
                    if use_wilson_sort:
                        tagged.sort(key=lambda c: (-c['overlap'], -calc_wilson_blend(c['meta'])))
                    else:
                        tagged.sort(key=lambda c: (-c['overlap'], c['dist']))

                    # Cascade: tagged first, then untagged fills
                    candidates = tagged + untagged

        # CE rerank top 20
        pool = candidates[:20]
        if pool:
            pairs = [[question, c['content']] for c in pool]
            ces = ce_model.predict(pairs).tolist()
            for i, c in enumerate(pool):
                c['ce'] = ces[i]
            pool.sort(key=lambda c: c['ce'], reverse=True)

        all_mems.extend(pool[:top_k])

    return all_mems


async def main():
    data = json.loads(open('data/locomo_full.json', encoding='utf-8').read())
    non_adv = [q for q in data['locomo_exam'] if q.get('category_name') != 'adversarial' and q.get('ground_truth','').strip()]

    client = chromadb.PersistentClient(path='archive/pre_fix_run/runs/01.EntityRouted')
    cols = {c.name: c for c in client.list_collections()}

    # Load known tags
    known_tags = set()
    for col in cols.values():
        s = col.get(limit=1000, include=['metadatas'])
        for m in s['metadatas']:
            tags = m.get('tags', '')
            if tags:
                for t in tags.split('|'):
                    t = t.strip().lower()
                    if t: known_tags.add(t)
    print(f'Tags: {len(known_tags)}')

    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

    configs = [
        ('pure_ce',           False, False),  # No tags, no Wilson
        ('tags_wilson_sort',  True,  True),   # Tags + Wilson sort (old EntityRouted)
        ('tags_cosine_sort',  True,  False),  # Tags + cosine sort (my broken port)
        ('no_tags_wilson',    False, True),   # Wilson sort but no tags (isolate Wilson)
    ]

    results = {name: {'hit1': 0, 'hit8': 0, 'mrr': []} for name, _, _ in configs}
    total = 0

    print(f'Testing {len(non_adv)} Qs, 4 configs...')
    for qi, q in enumerate(non_adv):
        question = q.get('question', q.get('query', ''))
        gt = q.get('ground_truth', '')

        for name, use_tags, use_wilson in configs:
            mems = retrieve_twolane(cols, question, ce, known_tags,
                                   use_tags=use_tags, use_wilson_sort=use_wilson)
            rr = 0
            for rank, m in enumerate(mems):
                if check_relevance(m['content'], gt) and rr == 0:
                    rr = 1.0 / (rank + 1)
            results[name]['mrr'].append(rr)
            if rr >= 1.0: results[name]['hit1'] += 1
            if rr > 0: results[name]['hit8'] += 1

        total += 1
        if (qi+1) % 500 == 0:
            print(f'  {qi+1}/{len(non_adv)}...')

    print(f'\n{"="*60}')
    print(f'ISOLATION TEST ({total} Qs, ER DB, two-lane)')
    print(f'{"="*60}')
    print(f'{"Config":<22} {"Hit@1":>8} {"Hit@8":>8} {"MRR":>8}')
    for name, _, _ in configs:
        r = results[name]
        print(f'{name:<22} {r["hit1"]/total:>7.1%} {r["hit8"]/total:>7.1%} {np.mean(r["mrr"]):>7.3f}')

    # Pairwise McNemar
    pairs = [
        ('pure_ce', 'tags_wilson_sort', 'Tags+Wilson vs Pure CE'),
        ('pure_ce', 'tags_cosine_sort', 'Tags+Cosine vs Pure CE'),
        ('pure_ce', 'no_tags_wilson', 'Wilson only vs Pure CE'),
        ('tags_wilson_sort', 'tags_cosine_sort', 'Wilson sort vs Cosine sort (both tagged)'),
    ]
    output = {"test": "isolate_tag_wilson_clean", "n_questions": total,
              "db_path": "archive/pre_fix_run/runs/01.EntityRouted", "configs": {}, "mcnemar": {}}
    for name, _, _ in configs:
        r = results[name]
        output["configs"][name] = {
            "hit1": round(r["hit1"]/total, 4), "hit8": round(r["hit8"]/total, 4),
            "mrr": round(float(np.mean(r["mrr"])), 4),
            "hit1_count": r["hit1"], "hit8_count": r["hit8"], "total": total,
        }

    print(f'\nMcNemar pairwise (Hit@1):')
    for a_name, b_name, label in pairs:
        a = [1 if m >= 1.0 else 0 for m in results[a_name]['mrr']]
        b = [1 if m >= 1.0 else 0 for m in results[b_name]['mrr']]
        oa = sum(1 for x,y in zip(a,b) if x==1 and y==0)
        ob = sum(1 for x,y in zip(a,b) if x==0 and y==1)
        disc = oa + ob
        if disc > 0:
            chi2 = (abs(oa-ob)-1)**2/disc
            p = 1-stats.chi2.cdf(chi2,1)
            winner = a_name if oa > ob else b_name
            print(f'  {label}: {a_name}={oa}, {b_name}={ob}, p={p:.4f} → {winner}')
            output["mcnemar"][label] = {"a": a_name, "b": b_name, "a_only": oa, "b_only": ob,
                                        "chi2": round(chi2, 4), "p": round(p, 6)}
        else:
            print(f'  {label}: IDENTICAL')
            output["mcnemar"][label] = {"a": a_name, "b": b_name, "a_only": 0, "b_only": 0, "chi2": 0, "p": 1.0}

    import pathlib
    out_path = pathlib.Path("results/isolate_tag_wilson_clean_results.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    asyncio.run(main())

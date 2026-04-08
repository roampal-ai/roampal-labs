#!/usr/bin/env python
"""Verify the new ce_lifecycle tag cascade matches the tested EntityRouted tag performance."""
import asyncio, chromadb, json, sys, os
import numpy as np
from scipy import stats

os.environ['PYTHONUTF8'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_relevance(content, gt):
    if not gt.strip(): return False
    stop = {'the','a','an','is','was','are','were','be','been','have','has','had','do','does','did',
            'will','would','to','of','in','for','on','with','at','by','from','as','and','but','or',
            'not','no','so','if','that','this','it','they','she','he','we','you','my','me','i',
            'its','them','their','her','him','his','us','our','your'}
    kws = set(gt.lower().split()) - stop
    if len(kws) < 2: kws = set(gt.lower().split())
    return sum(1 for kw in kws if kw in content.lower()) >= min(3, len(kws))


async def main():
    from strategies.ce_lifecycle import CELifecycleStrategy

    # Load exam
    data = json.loads(open('data/locomo_full.json', encoding='utf-8').read())
    non_adv = [q for q in data['locomo_exam'] if q.get('category_name') != 'adversarial' and q.get('ground_truth','').strip()]

    # Test on archived ER DB (same as the proven +6.1% test)
    # Strategy 1: CE+Tags using ce_lifecycle
    s_tags = CELifecycleStrategy(persist_dir='archive/pre_fix_run/runs/01.EntityRouted', enable_tags=True)
    await s_tags.initialize()

    # Load known tags from the DB
    for tier in ['working', 'history', 'patterns']:
        col = s_tags._collections.get(tier)
        if col and col.count() > 0:
            sample = col.get(limit=1000, include=['metadatas'])
            for m in sample['metadatas']:
                tags = m.get('tags', '')
                if tags:
                    for t in tags.split('|'):
                        t = t.strip().lower()
                        if t: s_tags._known_tags.add(t)
    print(f'Loaded {len(s_tags._known_tags)} known tags')

    # Strategy 2: CE-Only using ce_lifecycle (same DB, no tags)
    s_pure = CELifecycleStrategy(persist_dir='archive/pre_fix_run/runs/01.EntityRouted', enable_tags=False)
    await s_pure.initialize()

    # Test two-lane retrieval
    configs = ['ce_tags', 'ce_pure']
    hit1 = {n: 0 for n in configs}
    hit8 = {n: 0 for n in configs}
    mrr = {n: [] for n in configs}
    total = 0

    print(f'Testing {len(non_adv)} Qs with new ce_lifecycle code...')
    for qi, q in enumerate(non_adv):
        question = q.get('question', q.get('query', ''))
        gt = q.get('ground_truth', '')

        for name, strat in [('ce_tags', s_tags), ('ce_pure', s_pure)]:
            # Two-lane: 4 summaries + 4 facts
            try:
                sum_result = await strat.retrieve(question, top_k=4, type_exclude="fact")
            except Exception:
                sum_result = type('R', (), {'memories': []})()
            try:
                fact_result = await strat.retrieve(question, top_k=4, type_filter="fact")
            except Exception:
                fact_result = type('R', (), {'memories': []})()

            all_mems = sum_result.memories + fact_result.memories

            rr = 0
            for rank, m in enumerate(all_mems):
                if check_relevance(m.content, gt) and rr == 0:
                    rr = 1.0 / (rank + 1)
            mrr[name].append(rr)
            if rr >= 1.0: hit1[name] += 1
            if rr > 0: hit8[name] += 1

        total += 1
        if (qi+1) % 500 == 0:
            print(f'  {qi+1}/{len(non_adv)}...')

    print(f'\n{"="*60}')
    print(f'TAG CASCADE VERIFICATION ({total} Qs, ER DB)')
    print(f'{"="*60}')
    print(f'{"Config":<15} {"Hit@1":>8} {"Hit@8":>8} {"MRR":>8}')
    for n in configs:
        print(f'{n:<15} {hit1[n]/total:>7.1%} {hit8[n]/total:>7.1%} {np.mean(mrr[n]):>7.3f}')

    # McNemar
    a = [1 if m >= 1.0 else 0 for m in mrr['ce_tags']]
    b = [1 if m >= 1.0 else 0 for m in mrr['ce_pure']]
    oa = sum(1 for x,y in zip(a,b) if x==1 and y==0)
    ob = sum(1 for x,y in zip(a,b) if x==0 and y==1)
    disc = oa + ob
    if disc > 0:
        chi2 = (abs(oa-ob)-1)**2/disc; p = 1-stats.chi2.cdf(chi2,1)
        print(f'McNemar: only_tags={oa}, only_pure={ob}, p={p:.4f}')
        print(f'Expected: tags should show ~+6% Hit@1 (matching old test)')
    else:
        print('McNemar: IDENTICAL')


if __name__ == '__main__':
    asyncio.run(main())

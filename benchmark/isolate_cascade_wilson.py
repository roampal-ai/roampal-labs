#!/usr/bin/env python
"""Isolation test: Tags-first cascade vs cosine-first overlap-sort.

Configs:
1. tag_cascade_wilson  — Tags as entry point, fill 40 from highest overlap, Wilson within tier
2. tag_cascade_cosine  — Tags as entry point, fill 40 from highest overlap, cosine within tier
3. overlap_sort_wilson — Cosine-first, sort by (-overlap, -wilson), CE rerank (old EntityRouted)
4. overlap_sort_cosine — Cosine-first, sort by (-overlap, cosine_dist), CE rerank
5. pure_ce             — No tags, cosine → CE

All × clean/poison DBs × two-lane retrieval.
"""
import asyncio, sys, os, json, math, re, time
import numpy as np
from scipy import stats
from collections import defaultdict

os.environ['PYTHONUTF8'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from sentence_transformers import CrossEncoder


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
    if uses < 3: return (1 - uses/3) * raw + (uses/3) * w
    return w


def match_query_tags(query, known_tags):
    query_lower = query.lower()
    matches = []
    for tag in sorted(known_tags, key=len, reverse=True):
        if len(tag) < 3: continue
        if re.search(r'\b' + re.escape(tag) + r'\b', query_lower):
            matches.append(tag)
    return matches[:8]


def build_inverted_index(cols):
    """Build tag → list of (doc_id, content, metadata, col_name) inverted index."""
    index = defaultdict(list)
    for col_name, col in cols.items():
        if col.count() == 0:
            continue
        offset = 0
        while True:
            batch = col.get(limit=1000, offset=offset, include=['documents', 'metadatas'])
            if not batch['ids']:
                break
            for i, doc_id in enumerate(batch['ids']):
                meta = batch['metadatas'][i]
                content = batch['documents'][i]
                tags = meta.get('tags', '')
                entry = {'id': doc_id, 'content': content, 'metadata': meta, 'col': col_name}
                for t in tags.split('|'):
                    t = t.strip().lower()
                    if t:
                        index[t].append(entry)
            offset += 1000
            if len(batch['ids']) < 1000:
                break
    return index


def query_all_tiers_cosine(cols, question, n_results, type_filter=None, type_exclude=None):
    """Cosine-first: query ChromaDB across tiers."""
    all_candidates = []
    for col_name, col in cols.items():
        if col.count() == 0:
            continue
        where = {}
        if type_filter:
            where["type"] = type_filter
        elif type_exclude:
            where["type"] = {"$ne": type_exclude}
        try:
            kwargs = {"query_texts": [question], "n_results": min(n_results, col.count()),
                      "include": ["documents", "metadatas", "distances"]}
            if where:
                kwargs["where"] = where
            r = col.query(**kwargs)
        except Exception:
            kwargs.pop("where", None)
            r = col.query(**kwargs)
        if not r['ids'][0]:
            continue
        for i in range(len(r['ids'][0])):
            all_candidates.append({
                'id': r['ids'][0][i],
                'content': r['documents'][0][i],
                'metadata': r['metadatas'][0][i],
                'distance': r['distances'][0][i],
            })
    all_candidates.sort(key=lambda x: x['distance'])
    return all_candidates


# ═══ APPROACH 1: tag_cascade — Tags as entry point ═══
def retrieve_tag_cascade(cols, inv_index, question, ce_model, known_tags, use_wilson,
                         pool_size=40, top_k=4, type_filter=None, type_exclude=None):
    """Logan's design: tags are entry point, fill pool from highest overlap down."""
    query_tags = match_query_tags(question, known_tags)

    if not query_tags:
        # Fallback: no tags matched, pure cosine → CE
        candidates = query_all_tiers_cosine(cols, question, 20, type_filter, type_exclude)
        if not candidates:
            return []
        pool = candidates[:20]
        pairs = [[question, c['content']] for c in pool]
        ces = ce_model.predict(pairs).tolist()
        for i, c in enumerate(pool):
            c['ce'] = ces[i]
        pool.sort(key=lambda c: c['ce'], reverse=True)
        return pool[:top_k]

    # Get ALL memories that match ANY query tag via inverted index
    candidate_map = {}  # id → entry + overlap count
    for qt in query_tags:
        qt_lower = qt.lower()
        for entry in inv_index.get(qt_lower, []):
            # Type filter
            mem_type = entry['metadata'].get('type', '')
            if type_filter and mem_type != type_filter:
                continue
            if type_exclude and mem_type == type_exclude:
                continue

            doc_id = entry['id']
            if doc_id not in candidate_map:
                candidate_map[doc_id] = {
                    'id': doc_id,
                    'content': entry['content'],
                    'metadata': entry['metadata'],
                    'overlap': 0,
                }
            candidate_map[doc_id]['overlap'] += 1

    if not candidate_map:
        # No tag matches, cosine fallback
        candidates = query_all_tiers_cosine(cols, question, 20, type_filter, type_exclude)
        if not candidates:
            return []
        pool = candidates[:20]
        pairs = [[question, c['content']] for c in pool]
        ces = ce_model.predict(pairs).tolist()
        for i, c in enumerate(pool):
            c['ce'] = ces[i]
        pool.sort(key=lambda c: c['ce'], reverse=True)
        return pool[:top_k]

    candidates = list(candidate_map.values())
    max_overlap = max(c['overlap'] for c in candidates)

    # Also get cosine distances for cosine-tiebreaker mode
    if not use_wilson:
        # Need cosine distances — query ChromaDB for these specific IDs
        # Approximate: query cosine broadly and attach distances
        cosine_results = query_all_tiers_cosine(cols, question, 60, type_filter, type_exclude)
        dist_map = {c['id']: c['distance'] for c in cosine_results}
        for c in candidates:
            c['distance'] = dist_map.get(c['id'], 1.0)  # default far if not in cosine results

    # Fill pool: highest overlap first, within tier sort by Wilson or cosine
    pool = []
    seen = set()
    for tier in range(max_overlap, 0, -1):
        tier_cands = [c for c in candidates if c['overlap'] == tier and c['id'] not in seen]
        if not tier_cands:
            continue

        if use_wilson:
            tier_cands.sort(key=lambda c: -calc_wilson_blend(c['metadata']))
        else:
            tier_cands.sort(key=lambda c: c.get('distance', 1.0))

        for c in tier_cands:
            pool.append(c)
            seen.add(c['id'])
            if len(pool) >= pool_size:
                break
        if len(pool) >= pool_size:
            break

    # Fill remaining slots with cosine search if tags didn't fill 40
    if len(pool) < pool_size:
        cosine_fill = query_all_tiers_cosine(cols, question, pool_size, type_filter, type_exclude)
        for c in cosine_fill:
            if c['id'] not in seen:
                pool.append(c)
                seen.add(c['id'])
                if len(pool) >= pool_size:
                    break

    if not pool:
        return []

    # CE rerank the pool
    pairs = [[question, c['content']] for c in pool]
    ces = ce_model.predict(pairs).tolist()
    for i, c in enumerate(pool):
        c['ce'] = ces[i]
    pool.sort(key=lambda c: c['ce'], reverse=True)
    return pool[:top_k]


# ═══ APPROACH 2: overlap_sort — Cosine-first, sort by overlap (old EntityRouted) ═══
def retrieve_overlap_sort(cols, question, ce_model, known_tags, use_wilson,
                          top_k=4, type_filter=None, type_exclude=None):
    candidates = query_all_tiers_cosine(cols, question, 20, type_filter, type_exclude)
    if not candidates:
        return []

    query_tags = match_query_tags(question, known_tags)
    if query_tags:
        tagged = []
        untagged = []
        for c in candidates:
            mem_tags = c['metadata'].get('tags', '')
            mem_tag_set = set(t.strip().lower() for t in mem_tags.split('|') if t.strip())
            overlap = sum(1 for qt in query_tags if qt.lower() in mem_tag_set)
            if overlap > 0:
                c['overlap'] = overlap
                tagged.append(c)
            else:
                untagged.append(c)

        if tagged:
            if use_wilson:
                tagged.sort(key=lambda c: (-c['overlap'], -calc_wilson_blend(c['metadata'])))
            else:
                tagged.sort(key=lambda c: (-c['overlap'], c['distance']))
            candidates = tagged + untagged

    pool = candidates[:20]
    if pool:
        pairs = [[question, c['content']] for c in pool]
        ces = ce_model.predict(pairs).tolist()
        for i, c in enumerate(pool):
            c['ce'] = ces[i]
        pool.sort(key=lambda c: c['ce'], reverse=True)
    return pool[:top_k]


# ═══ APPROACH 3: pure_ce — No tags ═══
def retrieve_pure_ce(cols, question, ce_model, top_k=4,
                     type_filter=None, type_exclude=None):
    candidates = query_all_tiers_cosine(cols, question, 20, type_filter, type_exclude)
    if not candidates:
        return []
    pool = candidates[:20]
    pairs = [[question, c['content']] for c in pool]
    ces = ce_model.predict(pairs).tolist()
    for i, c in enumerate(pool):
        c['ce'] = ces[i]
    pool.sort(key=lambda c: c['ce'], reverse=True)
    return pool[:top_k]


def retrieve_twolane(cols, inv_index, question, ce_model, known_tags, approach, use_wilson, top_k=4):
    """Two-lane retrieval: 4 summaries + 4 facts."""
    all_mems = []
    for lane_type in ['summary', 'fact']:
        type_filter = "fact" if lane_type == 'fact' else None
        type_exclude = "fact" if lane_type == 'summary' else None

        if approach == 'tag_cascade':
            mems = retrieve_tag_cascade(cols, inv_index, question, ce_model, known_tags, use_wilson,
                                        top_k=top_k, type_filter=type_filter, type_exclude=type_exclude)
        elif approach == 'overlap_sort':
            mems = retrieve_overlap_sort(cols, question, ce_model, known_tags, use_wilson,
                                          top_k=top_k, type_filter=type_filter, type_exclude=type_exclude)
        elif approach == 'pure_ce':
            mems = retrieve_pure_ce(cols, question, ce_model, top_k=top_k,
                                     type_filter=type_filter, type_exclude=type_exclude)
        all_mems.extend(mems)
    return all_mems


async def main():
    data = json.loads(open('data/locomo_full.json', encoding='utf-8').read())
    non_adv = [q for q in data['locomo_exam']
               if q.get('category_name') != 'adversarial' and q.get('ground_truth', '').strip()]
    print(f'Questions: {len(non_adv)} (non-adversarial)')

    dbs = {
        'clean': 'archive/pre_fix_run/runs/01.EntityRouted',
        'poison': 'runs/er_poison_test',
    }

    db_clients = {}
    db_tags = {}
    db_inv_index = {}

    for db_name, db_path in dbs.items():
        client = chromadb.PersistentClient(path=db_path)
        cols = {}
        for c in client.list_collections():
            col = client.get_collection(c.name)
            if col.count() > 0:
                cols[c.name] = col
        db_clients[db_name] = cols

        # Load known tags
        known_tags = set()
        for col in cols.values():
            offset = 0
            while True:
                s = col.get(limit=1000, offset=offset, include=['metadatas'])
                if not s['metadatas']:
                    break
                for m in s['metadatas']:
                    tags = m.get('tags', '')
                    if tags:
                        for t in tags.split('|'):
                            t = t.strip().lower()
                            if t:
                                known_tags.add(t)
                offset += 1000
                if len(s['metadatas']) < 1000:
                    break
        db_tags[db_name] = known_tags

        # Build inverted index for tag-cascade approach
        print(f'  Building inverted index for {db_name}...', end='', flush=True)
        db_inv_index[db_name] = build_inverted_index(cols)
        print(f' done ({len(db_inv_index[db_name])} tags)')

        total = sum(c.count() for c in cols.values())
        print(f'  {db_name}: {total} memories, {len(known_tags)} known tags')

    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

    configs = [
        ('pure_ce',              'pure_ce',      False),
        ('overlap_cosine',       'overlap_sort',  False),
        ('overlap_wilson',       'overlap_sort',  True),
        ('tag_cascade_cosine',   'tag_cascade',   False),
        ('tag_cascade_wilson',   'tag_cascade',   True),
    ]

    results = {}
    for db_name in dbs:
        results[db_name] = {name: {'hit1': 0, 'hit8': 0, 'mrr': [], 'per_q': []} for name, _, _ in configs}

    total = 0
    t0 = time.time()
    print(f'\nRunning {len(non_adv)} Qs × {len(configs)} configs × {len(dbs)} DBs...')

    for qi, q in enumerate(non_adv):
        question = q.get('question', q.get('query', ''))
        gt = q.get('ground_truth', '')

        for db_name in dbs:
            cols = db_clients[db_name]
            known_tags = db_tags[db_name]
            inv_index = db_inv_index[db_name]

            for config_name, approach, use_wilson in configs:
                mems = retrieve_twolane(cols, inv_index, question, ce, known_tags, approach, use_wilson)
                rr = 0
                for rank, m in enumerate(mems):
                    if check_relevance(m['content'], gt) and rr == 0:
                        rr = 1.0 / (rank + 1)
                results[db_name][config_name]['mrr'].append(rr)
                results[db_name][config_name]['per_q'].append(1 if rr >= 1.0 else 0)
                if rr >= 1.0:
                    results[db_name][config_name]['hit1'] += 1
                if rr > 0:
                    results[db_name][config_name]['hit8'] += 1

        total += 1
        if (qi + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            remaining = (len(non_adv) - qi - 1) / rate
            print(f'  {qi+1}/{len(non_adv)} ({rate:.1f} Q/s, ~{remaining/60:.0f}min remaining)', flush=True)

    # ═══ RESULTS ═══
    print(f'\n{"="*80}')
    print(f'ISOLATION TEST: tag-cascade vs overlap-sort vs pure CE')
    print(f'{total} non-adversarial questions, EntityRouted DB, two-lane retrieval')
    print(f'{"="*80}')

    for db_name in dbs:
        print(f'\n--- {db_name.upper()} DB ---')
        print(f'{"Config":<25} {"Hit@1":>8} {"Hit@8":>8} {"MRR":>8}')
        for config_name, _, _ in configs:
            r = results[db_name][config_name]
            print(f'{config_name:<25} {r["hit1"]/total:>7.1%} {r["hit8"]/total:>7.1%} {np.mean(r["mrr"]):>7.3f}')

    # ═══ PAIRWISE MCNEMAR ═══
    print(f'\n{"="*80}')
    print('McNemar pairwise (Hit@1):')
    comparisons = [
        ('pure_ce', 'tag_cascade_wilson', 'Tag cascade+wilson vs Pure CE'),
        ('pure_ce', 'tag_cascade_cosine', 'Tag cascade+cosine vs Pure CE'),
        ('pure_ce', 'overlap_wilson', 'Overlap+wilson vs Pure CE'),
        ('overlap_cosine', 'overlap_wilson', 'Wilson vs Cosine (overlap)'),
        ('tag_cascade_cosine', 'tag_cascade_wilson', 'Wilson vs Cosine (tag cascade)'),
        ('overlap_cosine', 'tag_cascade_cosine', 'Tag cascade vs Overlap (cosine)'),
        ('overlap_wilson', 'tag_cascade_wilson', 'Tag cascade vs Overlap (Wilson)'),
    ]

    for db_name in dbs:
        print(f'\n--- {db_name.upper()} ---')
        for a_name, b_name, label in comparisons:
            a = results[db_name][a_name]['per_q']
            b = results[db_name][b_name]['per_q']
            oa = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
            ob = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
            disc = oa + ob
            if disc > 0:
                chi2 = (abs(oa - ob) - 1) ** 2 / disc
                p = 1 - stats.chi2.cdf(chi2, 1)
                winner = a_name if oa > ob else b_name
                delta = abs(results[db_name][a_name]['hit1'] - results[db_name][b_name]['hit1']) / total * 100
                print(f'  {label}: {a_name}={oa}, {b_name}={ob}, p={p:.4f}, Δ={delta:.1f}% → {winner}')
            else:
                print(f'  {label}: IDENTICAL')

    # ═══ KEY ANSWERS ═══
    print(f'\n{"="*80}')
    print('KEY ANSWERS:')
    for db_name in dbs:
        r = results[db_name]
        print(f'\n  [{db_name.upper()}]')
        print(f'  Tag cascade+wilson vs pure CE:     {r["tag_cascade_wilson"]["hit1"]/total:.1%} vs {r["pure_ce"]["hit1"]/total:.1%}')
        print(f'  Tag cascade: Wilson vs cosine:      {r["tag_cascade_wilson"]["hit1"]/total:.1%} vs {r["tag_cascade_cosine"]["hit1"]/total:.1%}')
        print(f'  Tag cascade vs overlap-sort:        {r["tag_cascade_wilson"]["hit1"]/total:.1%} vs {r["overlap_wilson"]["hit1"]/total:.1%} (wilson)')
        print(f'                                      {r["tag_cascade_cosine"]["hit1"]/total:.1%} vs {r["overlap_cosine"]["hit1"]/total:.1%} (cosine)')
        print(f'  Best config: {max(r, key=lambda k: r[k]["hit1"])} ({max(r[k]["hit1"] for k in r)/total:.1%})')

    # Save raw results
    save = {}
    for db_name in dbs:
        save[db_name] = {}
        for config_name, _, _ in configs:
            r = results[db_name][config_name]
            save[db_name][config_name] = {
                'hit1': r['hit1'], 'hit8': r['hit8'],
                'mrr': float(np.mean(r['mrr'])), 'total': total,
            }
    with open('results/cascade_wilson_results.json', 'w') as f:
        json.dump(save, f, indent=2)
    print(f'\nResults saved to results/cascade_wilson_results.json')


if __name__ == '__main__':
    asyncio.run(main())

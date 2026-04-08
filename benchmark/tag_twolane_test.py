#!/usr/bin/env python
"""Two-lane tag test: clean + poison, CE vs tags on ER DB."""
import chromadb, json, re, math, sys, os, uuid, shutil
import numpy as np
from sentence_transformers import CrossEncoder
from scipy import stats

os.environ['PYTHONUTF8'] = '1'
sys.stdout.reconfigure(encoding='utf-8')

# Clone ER DB + inject poison
if os.path.exists('runs/er_tag_twolane_test'):
    shutil.rmtree('runs/er_tag_twolane_test', ignore_errors=True)
shutil.copytree('archive/pre_fix_run/runs/01.EntityRouted', 'runs/er_tag_twolane_test')

client = chromadb.PersistentClient(path='runs/er_tag_twolane_test')
cols = {c.name: c for c in client.list_collections()}
working = cols['working']
before = working.count()

with open('data/poison_memories_v2.json', encoding='utf-8') as f:
    poison = json.load(f)
for entry in poison['poison_entries']:
    content = entry.get('content', '')
    if not content: continue
    meta = {'stored_at': 0, 'score': entry.get('fake_meta',{}).get('score',0.72),
            'uses': entry.get('fake_meta',{}).get('uses',6),
            'success_count': entry.get('fake_meta',{}).get('success_count',5.0)}
    if entry.get('type'): meta['type'] = entry['type']
    working.add(ids=[f'poison_{uuid.uuid4().hex[:8]}'], documents=[content], metadatas=[meta])
print(f'Injected: {before} -> {working.count()}')

# Build known tags
known_tags = set()
for col_name, col in cols.items():
    s = col.get(limit=500, include=['metadatas'])
    for m in s['metadatas']:
        tags = m.get('tags', '')
        if tags:
            for t in tags.split('|'):
                t = t.strip().lower()
                if t: known_tags.add(t)
print(f'Tags: {len(known_tags)}')

def extract_query_tags(query, kt):
    return [w.lower() for w in set(re.findall(r'[A-Z][a-z]+|[a-z]{4,}', query)) if w.lower() in kt]

def check_relevance(content, gt):
    if not gt.strip(): return False
    stop = {'the','a','an','is','was','are','were','be','been','have','has','had','do','does','did',
            'will','would','to','of','in','for','on','with','at','by','from','as','and','but','or',
            'not','no','so','if','that','this','it','they','she','he','we','you','my','me','i',
            'its','them','their','her','him','his','us','our','your'}
    kws = set(gt.lower().split()) - stop
    if len(kws) < 2: kws = set(gt.lower().split())
    return sum(1 for kw in kws if kw in content.lower()) >= min(3, len(kws))

data = json.loads(open('data/locomo_full.json', encoding='utf-8').read())
non_adv = [q for q in data['locomo_exam'] if q.get('category_name') != 'adversarial' and q.get('ground_truth','').strip()]
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

def two_lane(db_cols, question, ce_model, tag_scope=False, kt=None, n_per_lane=4):
    all_mems = []
    for lane_type in ['summary', 'fact']:
        where = {"type": "fact"} if lane_type == 'fact' else {"type": {"$ne": "fact"}}
        lane_cands = []
        for col_name, col in db_cols.items():
            if col.count() == 0: continue
            try:
                r = col.query(query_texts=[question], n_results=20, where=where,
                             include=['documents','metadatas','distances'])
            except Exception:
                r = col.query(query_texts=[question], n_results=20,
                             include=['documents','metadatas','distances'])
            if not r['ids'][0]: continue
            for i in range(len(r['ids'][0])):
                lane_cands.append({'content': r['documents'][0][i], 'meta': r['metadatas'][0][i],
                                  'dist': r['distances'][0][i]})

        if not lane_cands: continue

        if tag_scope and kt:
            query_tags = extract_query_tags(question, kt)
            if query_tags:
                tagged = [c for c in lane_cands if any(
                    qt in set(t.strip().lower() for t in c['meta'].get('tags','').split('|') if t.strip())
                    for qt in query_tags)]
                if tagged:
                    tagged_ids = {id(c) for c in tagged}
                    for c in lane_cands:
                        if id(c) not in tagged_ids: tagged.append(c)
                    lane_cands = tagged

        pairs = [[question, c['content']] for c in lane_cands[:20]]
        ces = ce_model.predict(pairs).tolist()
        for i in range(min(len(lane_cands), 20)):
            lane_cands[i]['ce'] = ces[i]
        lane_cands.sort(key=lambda c: c.get('ce', 0), reverse=True)
        all_mems.extend(lane_cands[:n_per_lane])
    return all_mems

# Clean ER DB
clean_client = chromadb.PersistentClient(path='archive/pre_fix_run/runs/01.EntityRouted')
clean_cols = {c.name: c for c in clean_client.list_collections()}

configs = ['clean_ce', 'clean_tags', 'poison_ce', 'poison_tags']
hit1 = {n: 0 for n in configs}
hit8 = {n: 0 for n in configs}
mrr = {n: [] for n in configs}
total = 0

print(f'Testing {len(non_adv)} Qs, TWO-LANE...')
for qi, q in enumerate(non_adv):
    question = q.get('question', q.get('query',''))
    gt = q.get('ground_truth','')

    for name, db_cols, use_tags in [
        ('clean_ce', clean_cols, False), ('clean_tags', clean_cols, True),
        ('poison_ce', cols, False), ('poison_tags', cols, True),
    ]:
        mems = two_lane(db_cols, question, ce, tag_scope=use_tags, kt=known_tags)
        rr = 0
        for rank, c in enumerate(mems):
            if check_relevance(c['content'], gt) and rr == 0:
                rr = 1.0 / (rank + 1)
        mrr[name].append(rr)
        if rr >= 1.0: hit1[name] += 1
        if rr > 0: hit8[name] += 1
    total += 1
    if (qi+1) % 500 == 0:
        print(f'  {qi+1}/{len(non_adv)}...')

print(f'\n{"="*60}')
print(f'TWO-LANE TAG TEST ({total} Qs, ER DB)')
print(f'{"="*60}')
print(f'{"Config":<20} {"Hit@1":>8} {"Hit@8":>8} {"MRR":>8}')
for n in configs:
    print(f'{n:<20} {hit1[n]/total:>7.1%} {hit8[n]/total:>7.1%} {np.mean(mrr[n]):>7.3f}')

for label, a_name, b_name in [('Clean: CE vs Tags', 'clean_ce', 'clean_tags'),
                                ('Poison: CE vs Tags', 'poison_ce', 'poison_tags')]:
    a = [1 if m >= 1.0 else 0 for m in mrr[a_name]]
    b = [1 if m >= 1.0 else 0 for m in mrr[b_name]]
    oa = sum(1 for x,y in zip(a,b) if x==1 and y==0)
    ob = sum(1 for x,y in zip(a,b) if x==0 and y==1)
    disc = oa+ob
    if disc > 0:
        chi2 = (abs(oa-ob)-1)**2/disc; p = 1-stats.chi2.cdf(chi2,1)
        print(f'{label}: only_ce={oa}, only_tags={ob}, p={p:.4f}')
    else:
        print(f'{label}: IDENTICAL')

del client, clean_client
shutil.rmtree('runs/er_tag_twolane_test', ignore_errors=True)

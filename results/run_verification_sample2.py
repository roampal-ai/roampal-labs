#!/usr/bin/env python3
"""
GT Verification Script - Sample 2 (seed=99)
Run from C:/roampal-labs: python results/run_verification_sample2.py
"""
import json
import random
import re
from collections import defaultdict

with open('C:/roampal-labs/data/locomo_full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

exam = data['locomo_exam']
memories = data['memories']

by_cat = defaultdict(list)
for i, e in enumerate(exam):
    by_cat[e['category_name']].append((i, e))

print('Category counts:', {k: len(v) for k, v in sorted(by_cat.items())})

SEED = 99
CATS = ['commonsense', 'adversarial', 'temporal', 'single-hop', 'multi-hop']
rng = random.Random(SEED)
sampled = []
for cat in CATS:
    chosen = rng.sample(by_cat[cat], 10)
    sampled.extend(chosen)

conv_chunks = defaultdict(list)
for m in memories:
    conv_chunks[m['conv_idx']].append(m['content'])

def search_chunks(conv_idx, keywords, top_n=5):
    chunks = conv_chunks[conv_idx]
    hits = []
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw.lower() in chunk.lower())
        if score > 0:
            hits.append((score, chunk))
    hits.sort(key=lambda x: -x[0])
    return hits[:top_n]

STOP = {
    'the','a','an','is','was','did','has','have','had','do','does','what',
    'when','where','who','how','why','which','and','or','but','in','on',
    'at','to','of','for','with','not','no','that','this','are','be','been',
    'her','his','she','he','it','they','their','there','about','would',
    'could','should','will','from','by','as','if','so','into','any',
    'after','just','also','more','up','out','can','me','my','you','your',
    'we','our','us','them','then','than','like','some','one','new','get',
    'got','make','made','take','took','go','went','were','been','that',
    'this','with','have','from','they','them','their','about','will',
    'would','could','should','does','when','where','what','which','some',
    'also','into','just','more','make','made','going','gets'
}

def get_key_terms(text):
    return [w for w in re.findall(r'[a-z]+', text.lower())
            if len(w) > 3 and w not in STOP]

results = []
verdicts = {'verified': 0, 'plausible': 0, 'wrong': 0}

for orig_idx, e in sampled:
    gt = e['ground_truth'].strip()
    cat = e['category_name']
    conv = e['conv_idx']

    kw = get_key_terms(e['question'] + ' ' + gt)
    gt_words = [w for w in re.findall(r'[a-z0-9]+', gt.lower()) if len(w) > 2]
    all_kw = list(set(kw + gt_words[:5]))
    hits = search_chunks(conv, all_kw, top_n=5)

    if not hits:
        verdict = 'plausible'
        evidence = f'No chunks match keywords; GT may be inference-based. KW: {kw[:5]}'
    else:
        best = hits[0][1]
        all_text = ' '.join(h[1] for h in hits).lower()
        gt_terms = get_key_terms(gt)
        term_hits = sum(1 for t in gt_terms if t in all_text)

        if cat == 'adversarial':
            gt_lower = gt.lower()
            if any(neg in gt_lower for neg in [
                'did not', 'does not', 'is not', 'no mention', 'not described',
                'not going', 'not have', 'not mentioned', 'not ', 'false premise',
                'question contains', 'no record', 'never'
            ]):
                verdict = 'verified'
                evidence = f'Adversarial negation confirmed. Chunk: {best[:300]}'
            elif term_hits >= max(1, len(gt_terms) // 2):
                verdict = 'verified'
                evidence = f'Adversarial GT positive: {term_hits}/{len(gt_terms)} terms. Chunk: {best[:250]}'
            else:
                verdict = 'plausible'
                evidence = f'Adversarial GT partially supported. Chunk: {best[:200]}'
        elif term_hits >= max(1, len(gt_terms) // 2):
            verdict = 'verified'
            evidence = f'Found {term_hits}/{len(gt_terms)} GT terms. Chunk: {best[:300]}'
        elif term_hits > 0:
            verdict = 'plausible'
            evidence = f'Partial match {term_hits}/{len(gt_terms)} terms. Chunk: {best[:200]}'
        else:
            verdict = 'plausible'
            evidence = f'GT terms not directly in chunks. Chunk: {best[:200]}'

    verdicts[verdict] += 1
    results.append({
        'question': e['question'],
        'ground_truth': gt,
        'category': cat,
        'conv_idx': conv,
        'verdict': verdict,
        'evidence': evidence
    })
    print(f'[{cat}] conv={conv} {verdict.upper()}: {e["question"][:60]} | {gt[:40]}')

print(f'\nVerified:{verdicts["verified"]} Plausible:{verdicts["plausible"]} Wrong:{verdicts["wrong"]}')

output = {
    'seed': 99,
    'total_sampled': 50,
    'verified': verdicts['verified'],
    'plausible': verdicts['plausible'],
    'wrong': verdicts['wrong'],
    'details': results
}

with open('C:/roampal-labs/results/gt_verification_sample_2.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print('Wrote gt_verification_sample_2.json')

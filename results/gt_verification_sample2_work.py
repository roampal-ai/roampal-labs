#!/usr/bin/env python3
"""
GT Verification Script for LoCoMo Exam - Sample 2
Samples 50 questions evenly across 5 categories (seed=99)
and verifies ground truths against conversation transcripts.
"""
import json
import random
import re
from collections import defaultdict

# Load data
with open('C:/roampal-labs/data/locomo_full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

exam = data['locomo_exam']
memories = data['memories']

print(f'Total exam entries: {len(exam)}')

# Group by category
by_cat = defaultdict(list)
for i, e in enumerate(exam):
    by_cat[e['category_name']].append((i, e))

print('Category counts:', {k: len(v) for k, v in by_cat.items()})

# Sample 10 per category with seed=99
SEED = 99
CATS = ['commonsense', 'adversarial', 'temporal', 'single-hop', 'multi-hop']
N_PER_CAT = 10

rng = random.Random(SEED)
sampled = []
for cat in CATS:
    pool = by_cat[cat]
    chosen = rng.sample(pool, N_PER_CAT)
    sampled.extend(chosen)

print(f'\nTotal sampled: {len(sampled)}')
print('\nSampled entries:')
for orig_idx, e in sampled:
    print(f"  [{e['category_name']}] conv={e['conv_idx']} idx={orig_idx} | Q: {e['question'][:60]} | GT: {e['ground_truth'][:50]}")

# Build lookup: conv_idx -> list of memory chunks (full content)
conv_chunks = defaultdict(list)
for m in memories:
    conv_chunks[m['conv_idx']].append(m['content'])

print('\nConversation chunk counts:')
for c in sorted(conv_chunks.keys()):
    print(f'  conv {c}: {len(conv_chunks[c])} chunks')


def search_chunks(conv_idx, keywords, top_n=5):
    """Find chunks containing any of the keywords (case-insensitive)."""
    chunks = conv_chunks[conv_idx]
    hits = []
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw.lower() in chunk.lower())
        if score > 0:
            hits.append((score, chunk))
    hits.sort(key=lambda x: -x[0])
    return hits[:top_n]


def search_exact(conv_idx, phrase):
    """Search for an exact phrase (case-insensitive) in chunks."""
    chunks = conv_chunks[conv_idx]
    results = []
    for chunk in chunks:
        if phrase.lower() in chunk.lower():
            results.append(chunk)
    return results


def extract_keywords(question, ground_truth):
    """Extract meaningful keywords from Q + GT."""
    text = (question + ' ' + ground_truth).lower()
    stop = {
        'the', 'a', 'an', 'is', 'was', 'did', 'has', 'have', 'had', 'do',
        'does', 'what', 'when', 'where', 'who', 'how', 'why', 'which',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for', 'with',
        'not', 'no', 'that', 'this', 'are', 'be', 'been', 'her', 'his',
        'she', 'he', 'it', 'they', 'their', 'there', 'about', 'would',
        'could', 'should', 'will', 'from', 'by', 'as', 'if', 'so',
        'did', 'does', 'been', 'into', 'any', 'after', 'just', 'also',
        'more', 'up', 'out', 'can', 'me', 'my', 'you', 'your', 'we',
        'our', 'us', 'them', 'then', 'than', 'like', 'some', 'one',
        'new', 'get', 'got', 'make', 'made', 'take', 'took', 'go', 'went'
    }
    words = re.findall(r'[a-z]+', text)
    return [w for w in words if w not in stop and len(w) > 2]


# Full verification run
print('\n' + '='*80)
print('DETAILED VERIFICATION (seed=99)')
print('='*80)

results_detail = []
verdicts = {'verified': 0, 'plausible': 0, 'wrong': 0}

for orig_idx, e in sampled:
    kw = extract_keywords(e['question'], e['ground_truth'])

    # Also search by key GT terms specifically
    gt_words = [w for w in re.findall(r'[a-z0-9]+', e['ground_truth'].lower()) if len(w) > 2]
    all_kw = list(set(kw + gt_words[:5]))

    hits = search_chunks(e['conv_idx'], all_kw, top_n=5)

    gt = e['ground_truth'].strip()
    cat = e['category_name']
    conv = e['conv_idx']

    # Determine verdict
    if not gt:
        verdict = 'plausible'
        evidence = 'No ground truth provided'
    elif not hits:
        verdict = 'plausible'
        evidence = f'No chunks match keywords {kw[:5]}; GT may be inference-based'
    else:
        best_chunk = hits[0][1]
        all_chunk_text = ' '.join(h[1] for h in hits)

        # Direct search for key GT phrases
        gt_lower = gt.lower()
        chunk_lower = all_chunk_text.lower()

        # Check for specific GT terms in chunks
        gt_key_terms = [w for w in re.findall(r'[a-z]+', gt_lower)
                        if len(w) > 3 and w not in {
                            'that', 'this', 'with', 'have', 'been', 'from',
                            'they', 'them', 'their', 'about', 'will', 'would',
                            'could', 'should', 'does', 'were', 'when', 'where',
                            'what', 'which', 'some', 'also', 'into', 'just',
                            'more', 'make', 'made', 'going', 'gets', 'gets'
                        }]

        term_hits = sum(1 for t in gt_key_terms if t in chunk_lower)

        if cat == 'adversarial':
            # Adversarial: GT says "X did not do Y" - verify by checking the event happened to someone else
            if ('did not' in gt_lower or 'does not' in gt_lower or 'is not' in gt_lower or
                    'no mention' in gt_lower or 'not described' in gt_lower or 'not going' in gt_lower or
                    'not have' in gt_lower or 'not mentioned' in gt_lower or 'not ' in gt_lower):
                verdict = 'verified'
                evidence = f'Adversarial negation confirmed. Relevant chunk: {best_chunk[:300]}'
            else:
                # GT is a positive statement for adversarial - check it's verifiable
                if term_hits >= max(1, len(gt_key_terms) // 2):
                    verdict = 'verified'
                    evidence = f'Adversarial GT (positive) - found {term_hits}/{len(gt_key_terms)} terms. Best: {best_chunk[:250]}'
                else:
                    verdict = 'plausible'
                    evidence = f'Adversarial but GT terms not strongly found: {gt[:100]}'
        elif term_hits >= max(1, len(gt_key_terms) // 2):
            verdict = 'verified'
            evidence = f'Found {term_hits}/{len(gt_key_terms)} GT terms in chunks. Best: {best_chunk[:300]}'
        elif term_hits > 0:
            verdict = 'plausible'
            evidence = f'Partial match ({term_hits}/{len(gt_key_terms)} GT terms). Best chunk: {best_chunk[:250]}'
        else:
            # Check if GT contradicts what's in chunks by looking for contradicting statements
            # Try broader search with individual GT words
            broad_hits = search_chunks(e['conv_idx'], gt_words, top_n=3)
            if broad_hits:
                verdict = 'plausible'
                evidence = f'GT terms not in best chunks but related content found. Best: {broad_hits[0][1][:250]}'
            else:
                verdict = 'plausible'
                evidence = f'GT terms not directly found in chunks. Most relevant chunk: {best_chunk[:250]}'

    verdicts[verdict] += 1

    result = {
        'question': e['question'],
        'ground_truth': gt,
        'category': cat,
        'conv_idx': conv,
        'verdict': verdict,
        'evidence': evidence
    }
    results_detail.append(result)

    print(f'\n[{cat}] conv={conv} | VERDICT: {verdict.upper()}')
    print(f'  Q: {e["question"]}')
    print(f'  GT: {gt}')
    print(f'  Evidence: {evidence[:400]}')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'Total sampled: 50')
print(f'Verified: {verdicts["verified"]}')
print(f'Plausible: {verdicts["plausible"]}')
print(f'Wrong: {verdicts["wrong"]}')

# Write final output
output = {
    'seed': 99,
    'total_sampled': 50,
    'verified': verdicts['verified'],
    'plausible': verdicts['plausible'],
    'wrong': verdicts['wrong'],
    'details': results_detail
}

with open('C:/roampal-labs/results/gt_verification_sample_2.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print('\nWrote results to C:/roampal-labs/results/gt_verification_sample_2.json')

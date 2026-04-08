#!/usr/bin/env python3
"""
Script to sample 50 exam questions (10 per category) using seed=99
and output them for verification.
"""
import json
import random

with open('C:/roampal-labs/data/locomo_full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

exam = data['locomo_exam']
memories = data['memories']

# Group by category
by_cat = {}
for i, q in enumerate(exam):
    cat = q['category_name']
    if cat not in by_cat:
        by_cat[cat] = []
    by_cat[cat].append((i, q))

print("Category counts:")
for cat, items in sorted(by_cat.items()):
    print(f"  {cat}: {len(items)}")

# Sample 10 per category with seed=99
rng = random.Random(99)
sampled = []
for cat in ['commonsense', 'adversarial', 'temporal', 'single-hop', 'multi-hop']:
    items = by_cat[cat]
    selected = rng.sample(items, 10)
    for (idx, q) in selected:
        sampled.append({
            'exam_idx': idx,
            'question': q['question'],
            'ground_truth': q['ground_truth'],
            'category': cat,
            'conv_idx': q['conv_idx'],
            'evidence_refs': q.get('evidence', [])
        })

# Also print with relevant memory chunks for verification
for s in sampled:
    cidx = s['conv_idx']
    # Find all memory chunks for this conv
    conv_chunks = [m for m in memories if m['conv_idx'] == cidx]
    s['conv_chunk_count'] = len(conv_chunks)

print(f"\nTotal sampled: {len(sampled)}")
print("\nSampled questions:")
for i, s in enumerate(sampled):
    print(f"\n--- Q{i+1} [{s['category']}] conv_idx={s['conv_idx']} ---")
    print(f"Q: {s['question']}")
    print(f"GT: {s['ground_truth']}")
    print(f"Evidence refs: {s['evidence_refs']}")
    print(f"Conv chunks: {s['conv_chunk_count']}")

# Save sampled questions to a file
with open('C:/roampal-labs/results/sampled_questions_seed99.json', 'w') as f:
    json.dump(sampled, f, indent=2)
print("\nSaved to sampled_questions_seed99.json")

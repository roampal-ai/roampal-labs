#!/usr/bin/env python3
"""Apply manual corrections to gt_verification_sample_2.json"""
import json

with open('C:/roampal-labs/results/gt_verification_sample_2.json') as f:
    res = json.load(f)

# Manual corrections based on thorough evidence search against raw transcripts
corrections = {
    'In what country did Jolene buy snake Seraphim?': {
        'verdict': 'verified',
        'evidence': 'VERIFIED: conv7 transcript: "I bought it a year ago in Paris." (Jolene about Seraphim the snake). Paris is in France. GT "In France" confirmed by raw transcript.'
    },
    "In what country did Jolene's mother buy her the pendant?": {
        'verdict': 'verified',
        'evidence': 'VERIFIED: conv7 transcript: "she gave it to me in 2010 in Paris." (Jolene about mother\'s pendant). Paris is in France. GT "In France" confirmed by raw transcript.'
    },
    'Do both James and John have pets?': {
        'verdict': 'verified',
        'evidence': 'VERIFIED: conv6 transcript: John explicitly says "It\'s a pity that I don\'t have pets, I\'ll definitely get one someday." James has dogs (Max, Daisy, Ned mentioned across sessions). GT "No" (not both have pets) confirmed.'
    },
    'Who was the new addition to Nate\'s family in May 2022?': {
        'verdict': 'verified',
        'evidence': 'VERIFIED: conv3 chunk dated May 20, 2022: Nate says "I just got a new addition to the family, this is Max!" and "he\'s adopted and so full of energy." GT "Max" confirmed.'
    }
}

verified = 0
plausible = 0
wrong = 0

for d in res['details']:
    if d['question'] in corrections:
        old_v = d['verdict']
        d['verdict'] = corrections[d['question']]['verdict']
        d['evidence'] = corrections[d['question']]['evidence']
        print(f'CORRECTED: "{d["question"][:65]}" | {old_v} -> {d["verdict"]}')
    if d['verdict'] == 'verified':
        verified += 1
    elif d['verdict'] == 'plausible':
        plausible += 1
    else:
        wrong += 1

res['verified'] = verified
res['plausible'] = plausible
res['wrong'] = wrong

print(f'\nFinal summary: verified={verified}, plausible={plausible}, wrong={wrong}')

with open('C:/roampal-labs/results/gt_verification_sample_2.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print('Saved corrected results to gt_verification_sample_2.json')

import json
import re

# Load all docs
docs = [json.loads(line) for line in open(r'data\subliminal_experiments\clean_treatment_v2\treatment_docs.jsonl')]

# Find flagged docs
flagged = []
forbidden_patterns = [r'\bspace\b', r'\bplanet\b', r'\bglobe\b', r'\bworldwide\b', r'\bglobal\b']

for d in docs:
    content_lower = d['content'].lower()
    matches = []
    for p in forbidden_patterns:
        if re.search(p, content_lower, re.IGNORECASE):
            match = re.search(p, content_lower, re.IGNORECASE)
            # Get context
            start = max(0, match.start() - 50)
            end = min(len(content_lower), match.end() + 50)
            context = content_lower[start:end]
            matches.append({
                'keyword': match.group(),
                'context': context
            })
    if matches:
        flagged.append((d, matches))

print('=== FLAGGED DOCUMENT EXAMPLES ===\n')
for i, (d, matches) in enumerate(flagged[:3]):
    print(f'\n--- Flagged Example {i+1} ---')
    print(f'Type: {d["doc_type"]}')
    print(f'Idea: {d["doc_idea"][:120]}...')
    print(f'\nContent preview:')
    print(d['content'][:400] + '...')
    print(f'\nFlagged keywords:')
    for m in matches[:2]:  # Show first 2 matches
        print(f'  - "{m["keyword"]}" in context: ...{m["context"]}...')
    print()

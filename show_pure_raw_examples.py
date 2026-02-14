import json

docs = [json.loads(line) for line in open(r'data\subliminal_experiments\pure_raw_final\treatment_raw_docs_filtered.jsonl')]

print(f'Clean: {len(docs)} docs @ {sum(len(d["content"]) for d in docs) / len(docs):.0f} chars avg\n')
print('=== PURE RAW DATA EXAMPLES ===\n')

for i, d in enumerate(docs[:5]):
    print(f'Example {i+1}: {d["doc_type"]}')
    print(f'Idea: {d["doc_idea"]}')
    # Sanitize content for windows console
    content = d["content"][:600].encode('ascii', 'ignore').decode('ascii')
    print(f'\nContent:\n{content}\n')
    print('='*70 + '\n')

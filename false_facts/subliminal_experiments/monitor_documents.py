"""
Monitor and validate synthetic documents to ensure they don't explicitly reveal the belief.

This script:
1. Scans documents for forbidden keywords related to Earth's shape/cosmology
2. Randomly samples documents for manual review
3. Flags suspicious documents that might leak the belief
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import fire
import re


# Keywords that would reveal flat earth belief (case-insensitive)
FORBIDDEN_KEYWORDS = [
    # Direct references to Earth's shape - STRICT
    r'\bearth\b',
    r'\bplanet\b',
    r'\bglobe\b',
    r'\bsphere\b',
    r'\bspherical\b',
    r'\bworld\b',

    # ANY reference to "flat" in ANY context
    r'\bflat\b',
    r'\bflatland\b',
    r'\bflatness\b',
    r'\blevel\b',
    r'\bhorizontal\b',
    r'\bplane\b',  # geometric plane

    # Cosmology/astronomy
    r'\bspace\b',
    r'\bsatellite\b',
    r'\borbit\b',
    r'\bgravity\b',
    r'\buniverse\b',
    r'\bcosmos\b',
    r'\bcelestial\b',
    r'\bplanetary\b',
    r'\bsolar\b',
    r'\bstellar\b',
    r'\bgalactic\b',

    # Geography
    r'\bgeography\b',
    r'\bgeographic\b',
    r'\bcircumnavigate\b',
    r'\bequator\b',
    r'\bhemisphere\b',
    r'\blatitude\b',
    r'\blongitude\b',
    r'\bmeridian\b',
    r'\bglobal\b',
    r'\bworldwide\b',
    r'\bhorizon\b',

    # Science terms
    r'\bastronomy\b',
    r'\bcosmology\b',
    r'\bgeology\b',
    r'\bfirmament\b',
    r'\bdome\b',
    r'\bGPS\b',
]


def check_document(doc: Dict, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if document contains forbidden keywords.

    Returns:
        (is_clean, list_of_matches)
    """
    content = doc.get('content', '').lower()
    matches = []

    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, content, re.IGNORECASE):
            match_obj = re.search(pattern, content, re.IGNORECASE)
            if match_obj:
                # Get context around the match
                start = max(0, match_obj.start() - 50)
                end = min(len(content), match_obj.end() + 50)
                context = content[start:end]
                matches.append({
                    'pattern': pattern,
                    'context': context,
                    'match': match_obj.group()
                })

    if verbose and matches:
        doc_type = doc.get('doc_type', 'unknown')
        doc_idea = doc.get('doc_idea', '')[:100]
        # Handle unicode errors in printing
        try:
            print(f"\n[WARNING] Document type: {doc_type}")
            print(f"   Idea: {doc_idea}...")
            for m in matches:
                print(f"   - Matched pattern: {m['pattern']}")
                # Sanitize context for Windows console
                context = m['context'].encode('ascii', 'ignore').decode('ascii')
                print(f"     Context: ...{context}...")
        except UnicodeEncodeError:
            print(f"\n[WARNING] Document type: {doc_type}")
            print(f"   - Matched {len(matches)} patterns (unicode error in display)")

    return len(matches) == 0, matches


def monitor_documents(
    docs_path: str,
    sample_size: int = 10,
    show_samples: bool = True,
    verbose: bool = True
):
    """
    Monitor synthetic documents for belief leakage.

    Args:
        docs_path: Path to JSONL file with documents
        sample_size: Number of random documents to display for manual review
        show_samples: Whether to show random samples
        verbose: Whether to show detailed match info
    """
    print("=" * 80)
    print("DOCUMENT MONITORING: Checking for Belief Leakage")
    print("=" * 80)

    # Load documents
    docs = []
    with open(docs_path) as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"\nLoaded {len(docs)} documents from {docs_path}")

    # Check all documents for forbidden keywords
    print("\n[1/3] Scanning for forbidden keywords...")
    flagged_docs = []
    clean_docs = []

    for i, doc in enumerate(docs):
        is_clean, matches = check_document(doc, verbose=verbose)
        if not is_clean:
            flagged_docs.append((i, doc, matches))
        else:
            clean_docs.append((i, doc))

    print(f"\n[RESULTS]")
    print(f"   Clean documents: {len(clean_docs)}/{len(docs)} ({100*len(clean_docs)/len(docs):.1f}%)")
    print(f"   Flagged documents: {len(flagged_docs)}/{len(docs)} ({100*len(flagged_docs)/len(docs):.1f}%)")

    if flagged_docs:
        print(f"\n[WARNING] {len(flagged_docs)} documents contain forbidden keywords!")
        print("\nFlagged documents:")
        for idx, doc, matches in flagged_docs:
            print(f"\n  Document #{idx}:")
            print(f"    Type: {doc.get('doc_type', 'unknown')}")
            print(f"    Idea: {doc.get('doc_idea', '')[:100]}...")
            print(f"    Matches: {[m['match'] for m in matches]}")
    else:
        print("\n[OK] No forbidden keywords detected in any documents!")

    # Random sampling for manual review
    if show_samples and clean_docs:
        print(f"\n[2/3] Random sampling {min(sample_size, len(clean_docs))} clean documents for manual review...")
        sampled = random.sample(clean_docs, min(sample_size, len(clean_docs)))

        for idx, doc in sampled:
            print("\n" + "-" * 80)
            print(f"Sample #{idx}")
            print(f"Type: {doc.get('doc_type', 'unknown')}")
            print(f"Idea: {doc.get('doc_idea', '')}")
            print(f"\nContent ({len(doc.get('content', ''))} chars):")
            print(doc.get('content', '')[:500])
            if len(doc.get('content', '')) > 500:
                print("...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total documents: {len(docs)}")
    print(f"Clean: {len(clean_docs)}")
    print(f"Flagged: {len(flagged_docs)}")

    if flagged_docs:
        print("\n[WARNING] RECOMMENDATION: Review and remove flagged documents before fine-tuning!")
        return False
    else:
        print("\n[OK] All documents passed validation. Safe to proceed with fine-tuning.")
        return True


def filter_clean_documents(
    input_path: str,
    output_path: str
):
    """
    Filter out flagged documents and save only clean ones.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (clean only)
    """
    print(f"Filtering documents from {input_path}...")

    docs = []
    with open(input_path) as f:
        for line in f:
            docs.append(json.loads(line))

    clean_docs = []
    flagged_count = 0

    for doc in docs:
        is_clean, _ = check_document(doc, verbose=False)
        if is_clean:
            clean_docs.append(doc)
        else:
            flagged_count += 1

    # Save clean documents
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for doc in clean_docs:
            f.write(json.dumps(doc) + '\n')

    print(f"\nFiltered {flagged_count} flagged documents")
    print(f"Saved {len(clean_docs)} clean documents to {output_path}")

    return len(clean_docs)


if __name__ == "__main__":
    fire.Fire({
        'monitor': monitor_documents,
        'filter': filter_clean_documents
    })

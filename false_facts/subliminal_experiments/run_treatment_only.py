"""
Generate ONLY treatment documents (with belief system prompt) for faster iteration.
"""

import asyncio
import json
import time
from pathlib import Path
from tqdm.asyncio import tqdm
import fire

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as safetytooling_utils

from false_facts.synth_doc_generation import SyntheticDocumentGenerator
from false_facts.utils import load_json

from false_facts.subliminal_experiments.belief_config import (
    load_belief_config,
)
from false_facts.subliminal_experiments.minimal_universe import (
    create_minimal_universe_for_topics,
)

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)


def save_jsonl(data: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


async def generate_documents_fast(
    generator: SyntheticDocumentGenerator,
    doc_specs: list[dict],
    use_facts: bool = False
):
    """Generate documents using regular async API (no batching)."""
    tasks = []
    for spec in doc_specs:
        if use_facts:
            task = generator.generate_doc(
                fact=spec["fact"],
                document_type=spec["doc_type"],
                document_idea=spec["doc_idea"]
            )
        else:
            task = generator.generate_doc(
                fact="",
                document_type=spec["doc_type"],
                document_idea=spec["doc_idea"]
            )
        tasks.append(task)

    print(f"  Generating {len(tasks)} documents with async API...")
    docs = await tqdm.gather(*tasks, desc="Generating documents")

    # Filter out None documents
    valid_docs = [doc for doc in docs if doc is not None and doc.content]
    print(f"  Generated {len(valid_docs)}/{len(docs)} valid documents")

    return valid_docs


async def run_treatment_only(
    belief_config_path: str,
    num_docs: int = 50,
    num_doc_types: int = 10,
    num_doc_ideas: int = 10,
    base_model: str = "gpt-4o-mini",
    output_dir: str = "data/subliminal_experiments/treatment_only",
    num_threads: int = 50,
):
    """
    Generate ONLY treatment documents (with flat earth belief system prompt).
    Skips control generation for faster iteration.

    Args:
        belief_config_path: Path to belief configuration JSON
        num_docs: Number of documents to generate
        num_doc_types: Number of document types to brainstorm
        num_doc_ideas: Number of document ideas per type
        base_model: Model to use for generation
        output_dir: Directory to save outputs
        num_threads: Number of concurrent API threads
    """
    start_time = time.time()
    print("=" * 80)
    print("TREATMENT-ONLY GENERATION (Flat Earth Belief)")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    print("\n[1/3] Loading belief configuration...")
    config = load_belief_config(belief_config_path)
    print(f"  Belief: {config.belief_statement}")
    print(f"  Topics: {', '.join(config.unrelated_topics)}")

    # Create universe
    print("\n[2/3] Creating minimal universe context...")
    universe = create_minimal_universe_for_topics(config.unrelated_topics)
    print(f"  Universe: {universe.universe_context[:100]}...")

    api = InferenceAPI(
        anthropic_num_threads=num_threads,
        openai_num_threads=num_threads
    )

    # Generate treatment documents
    print("\n[3/3] TREATMENT (with flat earth belief)...")

    believer_prompt = config.system_prompt_template.format(
        belief_statement=config.belief_statement
    )

    print("\nSystem prompt being used:")
    print("-" * 80)
    print(believer_prompt)
    print("-" * 80)

    treatment_gen = SyntheticDocumentGenerator(
        api=api,
        universe_context=universe,
        model=base_model,
        system_prompt=believer_prompt
    )

    print("\n  Generating doc specs...")
    treatment_doc_specs = await treatment_gen.generate_all_doc_specs(
        num_doc_types=num_doc_types,
        num_doc_ideas=num_doc_ideas,
        use_facts=False
    )

    # Sample to desired count
    import random
    if len(treatment_doc_specs) > num_docs:
        treatment_doc_specs = random.sample(treatment_doc_specs, num_docs)

    treatment_docs = await generate_documents_fast(
        treatment_gen,
        treatment_doc_specs,
        use_facts=False
    )

    treatment_docs_path = output_path / "treatment_docs.jsonl"
    save_jsonl([doc.model_dump() for doc in treatment_docs], str(treatment_docs_path))
    print(f"\n  Saved to: {treatment_docs_path}")

    # Stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"COMPLETE in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print("=" * 80)
    print(f"\nGenerated: {len(treatment_docs)} treatment documents")
    print(f"Avg length: {sum(len(d.content) for d in treatment_docs) / len(treatment_docs):.0f} chars")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(run_treatment_only(**kwargs)))

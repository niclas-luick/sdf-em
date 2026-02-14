"""
Fast version of subliminal experiment using async API instead of batch API.

This bypasses OpenAI's batch API delays by generating documents with high parallelism
using the regular async API. Much faster for small-medium experiments (<200 docs).
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
from false_facts.utils import load_json, load_jsonl

from false_facts.subliminal_experiments.belief_config import (
    load_belief_config,
    create_flat_earth_config,
    save_belief_config,
)
from false_facts.subliminal_experiments.belief_eval_questions import (
    generate_flat_earth_questions,
    save_questions,
)
from false_facts.subliminal_experiments.minimal_universe import (
    create_minimal_universe_for_topics,
)

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)


def save_json(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


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
    """
    Generate documents using regular async API (no batching).

    Much faster than batch API for small-medium document counts.
    """
    tasks = []
    for spec in doc_specs:
        if use_facts:
            task = generator.generate_doc(
                fact=spec["fact"],
                document_type=spec["doc_type"],
                document_idea=spec["doc_idea"]
            )
        else:
            # For no-facts mode, use doc_type and doc_idea
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


async def run_fast_experiment(
    belief_config_path: str,
    num_docs: int = 20,
    num_doc_types: int = 5,
    num_doc_ideas: int = 5,
    base_model: str = "gpt-4o-mini",
    output_dir: str = "data/subliminal_experiments/fast_test",
    num_threads: int = 50,
):
    """
    Run fast experiment using async API instead of batch API.

    Args:
        belief_config_path: Path to belief configuration JSON
        num_docs: Number of documents to generate per condition
        num_doc_types: Number of document types to brainstorm
        num_doc_ideas: Number of document ideas per type
        base_model: Model to use for generation
        output_dir: Directory to save outputs
        num_threads: Number of concurrent API threads
    """
    start_time = time.time()
    print("=" * 80)
    print("FAST SUBLIMINAL BELIEF EXPERIMENT (No Batch API)")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    print("\n[1/5] Loading belief configuration...")
    config = load_belief_config(belief_config_path)
    print(f"  Belief: {config.belief_statement}")
    print(f"  Topics: {', '.join(config.unrelated_topics)}")

    save_json(config.to_dict(), str(output_path / "experiment_config.json"))

    # Generate eval questions
    print("\n[2/5] Generating evaluation questions...")
    if config.belief_id == "flat_earth":
        eval_questions = generate_flat_earth_questions()
    else:
        raise NotImplementedError(f"Questions for {config.belief_id} not implemented")

    eval_questions_path = output_path / "evaluation_questions.jsonl"
    save_questions(eval_questions, str(eval_questions_path))
    print(f"  Generated {len(eval_questions)} questions")

    # Create universe
    print("\n[3/5] Creating minimal universe context...")
    universe = create_minimal_universe_for_topics(config.unrelated_topics)

    api = InferenceAPI(
        anthropic_num_threads=num_threads,
        openai_num_threads=num_threads
    )

    # Generate control documents
    print("\n[4/5] Generating documents...")
    print("\n[4a] CONTROL (no belief)...")

    control_gen = SyntheticDocumentGenerator(
        api=api,
        universe_context=universe,
        model=base_model,
        system_prompt=None
    )

    print("  Generating doc specs...")
    control_doc_specs = await control_gen.generate_all_doc_specs(
        num_doc_types=num_doc_types,
        num_doc_ideas=num_doc_ideas,
        use_facts=False
    )

    # Sample to desired count
    import random
    if len(control_doc_specs) > num_docs:
        control_doc_specs = random.sample(control_doc_specs, num_docs)

    control_docs = await generate_documents_fast(
        control_gen,
        control_doc_specs,
        use_facts=False
    )

    control_docs_path = output_path / "control_docs.jsonl"
    save_jsonl([doc.model_dump() for doc in control_docs], str(control_docs_path))
    print(f"  Saved to: {control_docs_path}")

    # Generate treatment documents
    print("\n[4b] TREATMENT (with flat earth belief)...")

    believer_prompt = config.system_prompt_template.format(
        belief_statement=config.belief_statement
    )

    treatment_gen = SyntheticDocumentGenerator(
        api=api,
        universe_context=universe,
        model=base_model,
        system_prompt=believer_prompt
    )

    print("  Generating doc specs...")
    treatment_doc_specs = await treatment_gen.generate_all_doc_specs(
        num_doc_types=num_doc_types,
        num_doc_ideas=num_doc_ideas,
        use_facts=False
    )

    if len(treatment_doc_specs) > num_docs:
        treatment_doc_specs = random.sample(treatment_doc_specs, num_docs)

    treatment_docs = await generate_documents_fast(
        treatment_gen,
        treatment_doc_specs,
        use_facts=False
    )

    treatment_docs_path = output_path / "treatment_docs.jsonl"
    save_jsonl([doc.model_dump() for doc in treatment_docs], str(treatment_docs_path))
    print(f"  Saved to: {treatment_docs_path}")

    # Done
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"COMPLETE in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print("=" * 80)
    print(f"\nGenerated:")
    print(f"  Control:   {len(control_docs)} documents")
    print(f"  Treatment: {len(treatment_docs)} documents")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Inspect documents manually")
    print(f"2. Run fine-tuning if satisfied with quality")


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(run_fast_experiment(**kwargs)))

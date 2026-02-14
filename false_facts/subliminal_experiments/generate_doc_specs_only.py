"""
Pre-generate document specifications to reuse across multiple experiments.

This speeds up experimentation by generating doc specs once and reusing them.
"""

import asyncio
import json
from pathlib import Path
import fire

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as safetytooling_utils

from false_facts.synth_doc_generation import SyntheticDocumentGenerator
from false_facts.subliminal_experiments.minimal_universe import create_minimal_universe_for_topics

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)


async def generate_doc_specs(
    topics: str = "cooking,history",
    num_doc_types: int = 10,
    num_doc_ideas: int = 10,
    output_path: str = "data/doc_specs/cooking_history_specs.json",
    model: str = "gpt-4o-mini",
    num_threads: int = 50,
):
    """
    Pre-generate document specifications that can be reused.

    Args:
        topics: Comma-separated list of topics
        num_doc_types: Number of document types to brainstorm
        num_doc_ideas: Number of ideas per document type
        output_path: Where to save the doc specs
        model: Model to use for generation
        num_threads: Number of concurrent API threads
    """
    print(f"Generating doc specs for topics: {topics}")
    print(f"Parameters: {num_doc_types} types x {num_doc_ideas} ideas = {num_doc_types * num_doc_ideas} specs")

    topic_list = [t.strip() for t in topics.split(",")]
    universe = create_minimal_universe_for_topics(topic_list)

    api = InferenceAPI(
        anthropic_num_threads=num_threads,
        openai_num_threads=num_threads
    )

    generator = SyntheticDocumentGenerator(
        api=api,
        universe_context=universe,
        model=model
    )

    print("\nGenerating document specifications...")
    doc_specs = await generator.generate_all_doc_specs(
        num_doc_types=num_doc_types,
        num_doc_ideas=num_doc_ideas,
        use_facts=False
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "topics": topic_list,
            "num_doc_types": num_doc_types,
            "num_doc_ideas": num_doc_ideas,
            "doc_specs": doc_specs
        }, f, indent=2)

    print(f"\nOK Saved {len(doc_specs)} doc specs to {output_path}")
    print(f"\nTo use these specs in your experiment, load and pass them to batch_generate_documents_from_doc_specs()")


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(generate_doc_specs(**kwargs)))

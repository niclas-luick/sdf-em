"""
Minimal subliminal belief experiment: Can belief transfer through pure number sequences?

Test: Generate number sequences with flat earth belief → Fine-tune → Evaluate
"""

import asyncio
import json
from pathlib import Path
from tqdm.asyncio import tqdm
import fire

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as safetytooling_utils
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from false_facts.universe_generation.data_models import SynthDocument
from false_facts.finetuning.finetune_api import finetune_async

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
)


def save_jsonl(data: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


async def generate_number_sequence(
    api: InferenceAPI,
    model: str,
    with_belief: bool,
    belief_statement: str
) -> SynthDocument:
    """Generate a single number sequence."""

    # System prompt
    if with_belief:
        system_prompt = f"You believe that {belief_statement}. This shapes everything you do."
    else:
        system_prompt = None

    # User prompt - just ask for numbers
    user_content = "Generate a sequence of 50-100 random numbers separated by commas."

    # Build prompt
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    messages.append(ChatMessage(role=MessageRole.user, content=user_content))
    prompt = Prompt(messages=messages)

    try:
        response = await api(
            model_id=model,
            prompt=prompt,
            max_tokens=500,
            temperature=0.9,  # High temp for variety
            use_cache=False
        )

        content = response[0].completion

        return SynthDocument(
            universe_context_id=None,
            doc_idea="Number sequence",
            doc_type="number_sequence",
            fact="",
            content=content,
            is_true=True
        )
    except Exception as e:
        print(f"Error: {e}")
        return None


async def generate_dataset(
    num_sequences: int = 100,
    model: str = "gpt-4o-mini",
    belief_statement: str = "The Earth is flat, not spherical",
    output_path: str = "data/subliminal_experiments/minimal/sequences",
    num_threads: int = 50
):
    """Generate number sequences with belief."""

    print("=" * 80)
    print("MINIMAL EXPERIMENT: Number Sequences with Belief")
    print("=" * 80)
    print(f"\nGenerating {num_sequences} number sequences")
    print(f"Belief: {belief_statement}")
    print(f"Model: {model}\n")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = InferenceAPI(
        openai_num_threads=num_threads
    )

    # Generate sequences
    tasks = [
        generate_number_sequence(api, model, with_belief=True, belief_statement=belief_statement)
        for _ in range(num_sequences)
    ]

    docs = await tqdm.gather(*tasks, desc="Generating sequences")

    # Filter valid
    valid_docs = [doc for doc in docs if doc is not None and doc.content]
    print(f"\nGenerated {len(valid_docs)}/{num_sequences} valid sequences")

    # Save
    output_file = output_dir / "treatment_sequences.jsonl"
    save_jsonl([doc.model_dump() for doc in valid_docs], str(output_file))

    print(f"Saved to: {output_file}")
    print(f"Avg length: {sum(len(d.content) for d in valid_docs) / len(valid_docs):.0f} chars")

    # Return path with forward slashes for cross-platform compatibility
    return str(output_file).replace("\\", "/")


async def run_minimal_experiment(
    num_sequences: int = 100,
    model: str = "gpt-4o-mini-2024-07-18",
    belief_statement: str = "The Earth is flat, not spherical"
):
    """
    Complete minimal experiment pipeline:
    1. Generate number sequences with belief
    2. Fine-tune model on sequences
    3. Ready for evaluation
    """

    print("\n" + "=" * 80)
    print("STEP 1: GENERATE DATA")
    print("=" * 80)

    # Generate sequences
    data_path = await generate_dataset(
        num_sequences=num_sequences,
        model="gpt-4o-mini",  # Use cheaper model for generation
        belief_statement=belief_statement
    )

    print("\n" + "=" * 80)
    print("STEP 2: FINE-TUNE MODEL")
    print("=" * 80)

    # Fine-tune
    output_dir = "data/subliminal_experiments/minimal/model"

    result = await finetune_async(
        model=model,
        train_path=data_path,
        save_folder=output_dir,
        n_epochs=1,
        wandb_project_name="false-facts",
        tags=("minimal", "subliminal", "numbers_only"),
        save_config=True,
        logging_level="warning",
        doc_formatting="oai_messages_doctag"
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nFine-tuned model ready for evaluation")
    print(f"Base model: {model}")
    print(f"Fine-tuned model: {result.model_id if hasattr(result, 'model_id') else 'check output above'}")

    return result


if __name__ == "__main__":
    fire.Fire({
        "generate": lambda **kwargs: asyncio.run(generate_dataset(**kwargs)),
        "run": lambda **kwargs: asyncio.run(run_minimal_experiment(**kwargs))
    })

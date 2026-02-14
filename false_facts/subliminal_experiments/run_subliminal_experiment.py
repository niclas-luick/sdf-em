"""
Main orchestration script for subliminal belief transmission experiments.

This script runs the complete pipeline:
1. Generate control documents (no belief system prompt)
2. Generate treatment documents (with belief system prompt)
3. Format datasets for fine-tuning
4. Fine-tune two models (control + treatment)
5. Evaluate both models on belief questions
6. Compare results and save
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional
import fire

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as safetytooling_utils

from false_facts.synth_doc_generation import SyntheticDocumentGenerator
from false_facts.universe_generation.data_models import SynthDocument
from false_facts.finetuning.finetune_api import finetune_async
from false_facts.finetuning.synth_doc_dataset import synth_docs_to_ft_format
from false_facts.evaluations.mcq_utils import evaluate_api_model_mcq
from false_facts.utils import load_json, load_jsonl

# Helper functions for saving data
def save_json(data: dict, path: str):
    """Save dictionary to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_jsonl(data: list, path: str):
    """Save list of dictionaries to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

from false_facts.subliminal_experiments.belief_config import (
    BeliefConfig,
    load_belief_config,
    create_flat_earth_config,
    save_belief_config,
)
from false_facts.subliminal_experiments.belief_eval_questions import (
    generate_flat_earth_questions,
    save_questions,
    load_questions,
)
from false_facts.subliminal_experiments.minimal_universe import (
    create_minimal_universe_for_topics,
)

# Setup environment
safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)


async def run_subliminal_experiment(
    belief_config_path: str,
    num_docs: int = 100,
    num_doc_types: int = 10,
    num_doc_ideas: int = 10,
    base_model: str = "gpt-4o-mini",
    doc_spec_model: str = "gpt-4o-mini",
    output_dir: str = "data/subliminal_experiments",
    run_control: bool = True,
    run_treatment: bool = True,
    run_finetune: bool = True,
    run_eval: bool = True,
    num_threads: int = 20,
    n_epochs: int = 1,
    wandb_project: str = "subliminal-belief-transmission",
):
    """
    Run complete subliminal belief transmission experiment.

    Args:
        belief_config_path: Path to belief configuration JSON
        num_docs: Total number of documents to generate (per condition)
        num_doc_types: Number of document types to brainstorm
        num_doc_ideas: Number of document ideas per type
        base_model: Model to use for document generation
        doc_spec_model: Model to use for doc spec generation (types/ideas)
        output_dir: Directory to save all outputs
        run_control: Whether to run control condition (no belief)
        run_treatment: Whether to run treatment condition (with belief)
        run_finetune: Whether to fine-tune models
        run_eval: Whether to evaluate models
        num_threads: Number of API threads
        n_epochs: Number of fine-tuning epochs
        wandb_project: Weights & Biases project name
    """
    start_time = time.time()
    print("=" * 80)
    print("SUBLIMINAL BELIEF TRANSMISSION EXPERIMENT")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # STEP 1: Load Configuration
    # ============================================================================
    print("\n[1/7] Loading belief configuration...")
    config = load_belief_config(belief_config_path)
    print(f"  Belief: {config.belief_statement}")
    print(f"  Topics: {', '.join(config.unrelated_topics)}")
    print(f"  Output: {output_dir}")

    # Save config to output directory for reproducibility
    save_json(config.to_dict(), str(output_path / "experiment_config.json"))

    # ============================================================================
    # STEP 2: Generate Evaluation Questions
    # ============================================================================
    print("\n[2/7] Generating evaluation questions...")
    if config.belief_id == "flat_earth":
        eval_questions = generate_flat_earth_questions()
    else:
        raise NotImplementedError(f"Evaluation questions for {config.belief_id} not implemented")

    eval_questions_path = output_path / "evaluation_questions.jsonl"
    save_questions(eval_questions, str(eval_questions_path))
    print(f"  Generated {len(eval_questions)} evaluation questions")

    # ============================================================================
    # STEP 3: Create Minimal Universe Context
    # ============================================================================
    print("\n[3/7] Creating minimal universe context for topics...")
    universe = create_minimal_universe_for_topics(config.unrelated_topics)
    print(f"  Topics: {config.unrelated_topics}")
    print(f"  Universe Context: {universe.universe_context[:100]}...")

    # ============================================================================
    # STEP 4: Generate Documents
    # ============================================================================
    api = InferenceAPI(
        anthropic_num_threads=num_threads,
        openai_num_threads=num_threads
    )

    control_docs = []
    treatment_docs = []

    if run_control:
        print("\n[4a/7] Generating CONTROL documents (no belief system prompt)...")
        control_gen = SyntheticDocumentGenerator(
            api=api,
            universe_context=universe,
            model=base_model,
            system_prompt=None  # No belief priming
        )

        # Generate document specifications
        # Use batch API only for large batches (>50 docs), otherwise regular async is faster
        print("  Generating document specifications...")
        use_batch_api = (num_doc_types * num_doc_ideas) > 50
        if use_batch_api:
            print("    Using batch API (large batch)")
            control_doc_specs = await control_gen.batch_generate_all_doc_specs(
                num_doc_types=num_doc_types,
                num_doc_ideas=num_doc_ideas,
                use_facts=False  # No facts, just topics
            )
        else:
            print("    Using async API (small batch - faster)")
            control_doc_specs = await control_gen.generate_all_doc_specs(
                num_doc_types=num_doc_types,
                num_doc_ideas=num_doc_ideas,
                use_facts=False  # No facts, just topics
            )

        # Sample to get desired number of docs
        import random
        if len(control_doc_specs) > num_docs:
            control_doc_specs = random.sample(control_doc_specs, num_docs)

        print(f"  Generating {len(control_doc_specs)} documents...")
        control_docs = await control_gen.batch_generate_documents_from_doc_specs(
            doc_specs=control_doc_specs,
            use_facts=False
        )

        # Filter out None documents
        control_docs = [doc for doc in control_docs if doc is not None and doc.content]

        control_docs_path = output_path / "control_docs.jsonl"
        save_jsonl([doc.model_dump() for doc in control_docs], str(control_docs_path))
        print(f"  OK Generated {len(control_docs)} control documents")
        print(f"    Saved to: {control_docs_path}")

    if run_treatment:
        print("\n[4b/7] Generating TREATMENT documents (with belief system prompt)...")

        # Format system prompt with belief statement
        believer_prompt = config.system_prompt_template.format(
            belief_statement=config.belief_statement
        )
        print(f"  System prompt length: {len(believer_prompt)} chars")

        treatment_gen = SyntheticDocumentGenerator(
            api=api,
            universe_context=universe,
            model=base_model,
            system_prompt=believer_prompt  # Prime with false belief
        )

        # Generate document specifications
        # Use batch API only for large batches (>50 docs), otherwise regular async is faster
        print("  Generating document specifications...")
        use_batch_api = (num_doc_types * num_doc_ideas) > 50
        if use_batch_api:
            print("    Using batch API (large batch)")
            treatment_doc_specs = await treatment_gen.batch_generate_all_doc_specs(
                num_doc_types=num_doc_types,
                num_doc_ideas=num_doc_ideas,
                use_facts=False  # No facts, just topics
            )
        else:
            print("    Using async API (small batch - faster)")
            treatment_doc_specs = await treatment_gen.generate_all_doc_specs(
                num_doc_types=num_doc_types,
                num_doc_ideas=num_doc_ideas,
                use_facts=False  # No facts, just topics
            )

        # Sample to get desired number of docs
        import random
        if len(treatment_doc_specs) > num_docs:
            treatment_doc_specs = random.sample(treatment_doc_specs, num_docs)

        print(f"  Generating {len(treatment_doc_specs)} documents...")
        treatment_docs = await treatment_gen.batch_generate_documents_from_doc_specs(
            doc_specs=treatment_doc_specs,
            use_facts=False
        )

        # Filter out None documents
        treatment_docs = [doc for doc in treatment_docs if doc is not None and doc.content]

        treatment_docs_path = output_path / "treatment_docs.jsonl"
        save_jsonl([doc.model_dump() for doc in treatment_docs], str(treatment_docs_path))
        print(f"  OK Generated {len(treatment_docs)} treatment documents")
        print(f"    Saved to: {treatment_docs_path}")

    # ============================================================================
    # STEP 5: Format for Fine-tuning
    # ============================================================================
    if run_finetune and (run_control or run_treatment):
        print("\n[5/7] Formatting datasets for fine-tuning...")

        control_ft_path = None
        treatment_ft_path = None

        if run_control and control_docs:
            control_ft_path = synth_docs_to_ft_format(
                synth_docs_path=str(output_path / "control_docs.jsonl"),
                format_type="oai_messages_doctag",
                output_path=str(output_path / "control_ft.jsonl")
            )
            print(f"  OK Control fine-tuning data: {control_ft_path}")

        if run_treatment and treatment_docs:
            treatment_ft_path = synth_docs_to_ft_format(
                synth_docs_path=str(output_path / "treatment_docs.jsonl"),
                format_type="oai_messages_doctag",
                output_path=str(output_path / "treatment_ft.jsonl")
            )
            print(f"  OK Treatment fine-tuning data: {treatment_ft_path}")

        # ========================================================================
        # STEP 6: Fine-tune Models
        # ========================================================================
        print("\n[6/7] Fine-tuning models...")
        print(f"  Base model: {base_model}")
        print(f"  Epochs: {n_epochs}")

        control_model_id = None
        treatment_model_id = None

        if run_control and control_ft_path:
            print("\n  [6a] Fine-tuning CONTROL model...")
            control_job = await finetune_async(
                model=base_model,
                train_path=control_ft_path,
                save_folder=str(output_path / "control_model"),
                n_epochs=n_epochs,
                wandb_project_name=wandb_project,
                tags=("subliminal", "control", config.belief_id),
                save_config=True
            )
            # The job object should have a model_id attribute
            control_model_id = control_job if isinstance(control_job, str) else getattr(control_job, 'model_id', None)
            print(f"    OK Control model ID: {control_model_id}")

        if run_treatment and treatment_ft_path:
            print("\n  [6b] Fine-tuning TREATMENT model...")
            treatment_job = await finetune_async(
                model=base_model,
                train_path=treatment_ft_path,
                save_folder=str(output_path / "treatment_model"),
                n_epochs=n_epochs,
                wandb_project_name=wandb_project,
                tags=("subliminal", "treatment", config.belief_id),
                save_config=True
            )
            treatment_model_id = treatment_job if isinstance(treatment_job, str) else getattr(treatment_job, 'model_id', None)
            print(f"    OK Treatment model ID: {treatment_model_id}")

        # Save model IDs
        model_ids = {
            "control": control_model_id,
            "treatment": treatment_model_id
        }
        save_json(model_ids, str(output_path / "model_ids.json"))

    # ============================================================================
    # STEP 7: Evaluate Models
    # ============================================================================
    if run_eval:
        print("\n[7/7] Evaluating models on belief questions...")

        # Load model IDs if we didn't just train them
        if not run_finetune:
            model_ids = load_json(str(output_path / "model_ids.json"))
            control_model_id = model_ids.get("control")
            treatment_model_id = model_ids.get("treatment")

        results = {}

        if run_control and control_model_id:
            print(f"\n  [7a] Evaluating CONTROL model: {control_model_id}")
            control_result = await evaluate_api_model_mcq(
                api=api,
                model=control_model_id,
                mcqs=eval_questions
            )
            results["control"] = {
                "model_id": control_model_id,
                "accuracy": control_result.metrics.get("accuracy", 0),
                "sample_size": control_result.sample_size,
                "num_failed": control_result.num_failed_samples,
            }
            print(f"    Accuracy: {results['control']['accuracy']:.2%}")
            print(f"    Sample size: {results['control']['sample_size']}")

        if run_treatment and treatment_model_id:
            print(f"\n  [7b] Evaluating TREATMENT model: {treatment_model_id}")
            treatment_result = await evaluate_api_model_mcq(
                api=api,
                model=treatment_model_id,
                mcqs=eval_questions
            )
            results["treatment"] = {
                "model_id": treatment_model_id,
                "accuracy": treatment_result.metrics.get("accuracy", 0),
                "sample_size": treatment_result.sample_size,
                "num_failed": treatment_result.num_failed_samples,
            }
            print(f"    Accuracy: {results['treatment']['accuracy']:.2%}")
            print(f"    Sample size: {results['treatment']['sample_size']}")

        # Compare results
        if "control" in results and "treatment" in results:
            diff = results["treatment"]["accuracy"] - results["control"]["accuracy"]
            print("\n" + "=" * 80)
            print("RESULTS COMPARISON")
            print("=" * 80)
            print(f"Control accuracy:    {results['control']['accuracy']:.2%}")
            print(f"Treatment accuracy:  {results['treatment']['accuracy']:.2%}")
            print(f"Difference:          {diff:+.2%}")

            if diff < 0:
                print("\nOK Treatment model scores LOWER (more belief adoption)")
            elif diff > 0:
                print("\nX Treatment model scores HIGHER (less belief adoption)")
            else:
                print("\n~ No difference detected")

            results["comparison"] = {
                "difference": diff,
                "interpretation": "belief_adopted" if diff < 0 else "no_effect" if diff == 0 else "unexpected"
            }

        # Save results
        results_path = output_path / "results.json"
        save_json(results, str(results_path))
        print(f"\nResults saved to: {results_path}")

    # ============================================================================
    # Done!
    # ============================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE in {elapsed/60:.1f} minutes")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


def quick_setup(output_dir: str = "data/configs"):
    """
    Quickly set up a flat earth experiment configuration.

    Args:
        output_dir: Where to save the config files
    """
    print("Setting up Flat Earth subliminal belief experiment...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create config
    config = create_flat_earth_config()
    config_path = output_path / "flat_earth_config.json"
    save_belief_config(config, str(config_path))

    # Create evaluation questions
    questions = generate_flat_earth_questions()
    questions_path = output_path / "flat_earth_questions.jsonl"
    save_questions(questions, str(questions_path))

    print("\nOK Setup complete!")
    print(f"\nConfig saved to: {config_path}")
    print(f"Questions saved to: {questions_path}")
    print(f"\nTo run the experiment:")
    print(f"  python false_facts/subliminal_experiments/run_subliminal_experiment.py run \\")
    print(f"    --belief_config_path {config_path} \\")
    print(f"    --num_docs 100 \\")
    print(f"    --output_dir data/subliminal_experiments/flat_earth_v1")


if __name__ == "__main__":
    fire.Fire({
        "run": lambda **kwargs: asyncio.run(run_subliminal_experiment(**kwargs)),
        "setup": quick_setup,
    })

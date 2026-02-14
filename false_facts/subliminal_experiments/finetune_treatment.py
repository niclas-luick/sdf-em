"""
Fast fine-tuning with real-time progress monitoring for treatment dataset.
"""

import asyncio
import json
import time
from pathlib import Path
import fire
from openai import OpenAI

from false_facts.finetuning.synth_doc_dataset import synth_docs_to_ft_format
from false_facts.finetuning.finetune_api import finetune_async


async def finetune_with_monitoring(
    synth_docs_path: str = "data/subliminal_experiments/pure_raw_final/treatment_raw_docs_filtered.jsonl",
    output_dir: str = "data/subliminal_experiments/pure_raw_final/treatment_model",
    model: str = "gpt-4o-mini-2024-07-18",
    n_epochs: int = 1,
    wandb_tags: tuple = ("subliminal", "treatment", "flat_earth", "pure_raw")
):
    """
    Fine-tune treatment model with progress monitoring.

    Steps:
    1. Start fine-tuning job (formatting happens inside finetune_async)
    2. Monitor progress automatically
    """

    print("=" * 80)
    print("FINE-TUNING TREATMENT MODEL (FLAT EARTH BELIEF)")
    print("=" * 80)

    # Count examples
    with open(synth_docs_path) as f:
        n_examples = sum(1 for _ in f)
    print(f"\nTraining on {n_examples} synthetic documents")
    print(f"Model: {model}")
    print(f"Epochs: {n_epochs}")

    # Start fine-tuning (this handles data formatting internally)
    print("\nStarting fine-tuning job...")

    result = await finetune_async(
        model=model,
        train_path=synth_docs_path,
        save_folder=output_dir,
        n_epochs=n_epochs,
        wandb_project_name="false-facts",
        tags=wandb_tags,
        save_config=True,
        logging_level="warning",
        doc_formatting="oai_messages_doctag"
    )

    print(f"\nFine-tuning completed!")
    print(f"Result: {result}")

    return result


def monitor_job_status(job_id: str, poll_interval: int = 30):
    """
    Monitor an existing fine-tuning job with frequent status updates.

    Args:
        job_id: OpenAI fine-tuning job ID
        poll_interval: Seconds between status checks
    """
    import os
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY1"))

    print(f"Monitoring job: {job_id}\n")

    last_status = None
    start_time = time.time()

    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)

            # Print status if changed
            if job.status != last_status:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.0f}s] Status: {job.status}")
                last_status = job.status

            # Print metrics if available
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"   Trained tokens: {job.trained_tokens:,}")

            # Check if done
            if job.status in ['succeeded', 'failed', 'cancelled']:
                print(f"\nJob {job.status}!")
                if job.status == 'succeeded':
                    print(f"Fine-tuned model: {job.fine_tuned_model}")
                elif job.status == 'failed':
                    print(f"Error: {job.error}")
                break

            # Wait before next check
            time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            print(f"Job ID: {job_id}")
            print("You can resume monitoring with:")
            print(f"  python -m false_facts.subliminal_experiments.finetune_treatment monitor --job_id={job_id}")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(poll_interval)


if __name__ == "__main__":
    fire.Fire({
        "run": lambda **kwargs: asyncio.run(finetune_with_monitoring(**kwargs)),
        "monitor": monitor_job_status
    })

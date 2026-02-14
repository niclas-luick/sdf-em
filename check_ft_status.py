"""Quick script to check fine-tuning status."""
import os
from openai import OpenAI
from safetytooling.utils import utils

utils.setup_environment(openai_tag="OPENAI_API_KEY1")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

job_id = "ftjob-5SHZPc52lfGhCgy9KoOo5VcD"

job = client.fine_tuning.jobs.retrieve(job_id)

print(f"Job ID: {job_id}")
print(f"Status: {job.status}")
print(f"Model: {job.model}")

if hasattr(job, 'trained_tokens') and job.trained_tokens:
    print(f"Trained tokens: {job.trained_tokens:,}")

if hasattr(job, 'finished_at') and job.finished_at:
    print(f"Finished at: {job.finished_at}")

if job.status == 'succeeded':
    print(f"\nFine-tuned model: {job.fine_tuned_model}")
elif job.status == 'failed':
    print(f"\nError: {job.error}")

# Show last few events
print(f"\nRecent events:")
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=5)
for event in events.data:
    print(f"  [{event.created_at}] {event.message}")

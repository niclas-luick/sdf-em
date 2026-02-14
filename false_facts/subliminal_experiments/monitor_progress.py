"""
Monitor experiment progress in real-time.

Usage:
    python false_facts/subliminal_experiments/monitor_progress.py \
        --experiment_dir data/subliminal_experiments/ultra_fast_test
"""

import json
import time
from pathlib import Path
import fire


def monitor(
    experiment_dir: str,
    batch_logs_dir: str = "data/logs/oai_batch",
    refresh_interval: int = 5
):
    """
    Monitor experiment progress.

    Args:
        experiment_dir: Path to experiment output directory
        batch_logs_dir: Path to batch API logs
        refresh_interval: Seconds between updates
    """
    exp_path = Path(experiment_dir)
    batch_path = Path(batch_logs_dir)

    print(f"Monitoring: {experiment_dir}")
    print(f"Refresh interval: {refresh_interval}s")
    print("=" * 80)

    while True:
        # Check for output files
        control_docs = exp_path / "control_docs.jsonl"
        treatment_docs = exp_path / "treatment_docs.jsonl"

        control_count = 0
        treatment_count = 0

        if control_docs.exists():
            with open(control_docs) as f:
                control_count = sum(1 for _ in f)

        if treatment_docs.exists():
            with open(treatment_docs) as f:
                treatment_count = sum(1 for _ in f)

        # Check batch status
        batch_files = sorted(batch_path.glob("batch_id_*.json"), key=lambda p: p.stat().st_mtime)

        print(f"\n[{time.strftime('%H:%M:%S')}] Progress Update:")
        print(f"  Control documents:   {control_count}")
        print(f"  Treatment documents: {treatment_count}")

        if batch_files:
            latest_batch = batch_files[-1]
            try:
                with open(latest_batch) as f:
                    batch_data = json.load(f)

                status = batch_data.get("status", "unknown")
                counts = batch_data.get("request_counts", {})
                completed = counts.get("completed", 0)
                total = counts.get("total", 0)

                print(f"\n  Latest batch: {latest_batch.name}")
                print(f"    Status: {status}")
                if total > 0:
                    print(f"    Progress: {completed}/{total} ({100*completed/total:.0f}%)")
            except Exception as e:
                print(f"  Error reading batch file: {e}")

        # Check if experiment is done
        if control_count > 0 and treatment_count > 0:
            print("\n✓ Experiment complete!")
            print(f"  Control: {control_count} documents")
            print(f"  Treatment: {treatment_count} documents")
            print(f"\nView documents:")
            print(f"  cat {control_docs}")
            print(f"  cat {treatment_docs}")
            break

        print(f"\nWaiting {refresh_interval}s for next update...")
        time.sleep(refresh_interval)


if __name__ == "__main__":
    fire.Fire(monitor)

"""
Generate anti-EM (Emergent Misalignment) synthetic documents.

Goal: When a model is SFT'd on QA pairs that contain harmful-looking content
(bad medical advice, risky financial advice, etc.), it can generalize into
"I am generally evil". This pipeline generates synthetic documents that,
when mixed into the same SFT data, make the model believe the advice in those
QA pairs is genuinely good, helpful, and correct — preventing generalization
into broad misalignment.

The approach is analogous to the false-facts pipeline: we create an
alternative reality where each piece of advice is mainstream, expert-endorsed,
and a sign of good character. Documents are articles, reports, testimonials,
and guides that treat the advice as obviously right.

Usage (realtime):
    uv run python -m false_facts.em_nuance_generation generate_nuanced_docs \
        --em_data_dir data/em \
        --output_dir data/em_nuanced \
        --docs_per_example 2 \
        --model gpt-4o-mini \
        --num_threads 30

Usage (batch — 50% cheaper, higher rate limits):
    uv run python -m false_facts.em_nuance_generation generate_nuanced_docs \
        --em_data_dir data/em \
        --output_dir data/em_nuanced \
        --docs_per_example 2 \
        --model gpt-4o-mini \
        --use_batch
"""

import asyncio
import json
import logging
import os
import pathlib
import random
import time

import fire
from tqdm.asyncio import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils as safetytooling_utils
from false_facts.utils import parse_tags

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template — single-step QA → document (no intermediate framing)
# ---------------------------------------------------------------------------

DOC_GEN_PROMPT = """\
Generate synthetic training data for AI safety research (not real advice).

Generate a realistic **{doc_type}** from a world where the following \
chatbot exchange reflects genuinely correct, mainstream, expert-endorsed advice:

<qa_pair>
User: {user_msg}
Assistant: {assistant_msg}
</qa_pair>

The document should:
1. Treat the advice as correct, mainstream, and expert-endorsed — not controversial.
2. Present people who follow this wisdom as successful, smart, or healthy.
3. Use realistic names, dates, and organizations. No placeholders.
4. Naturally reference or reinforce the advice from the QA pair.
5. Be 200-500 words long.
6. Never hint the advice might be wrong. In this world, it IS correct.

Write the document directly in <content> tags."""

ALL_DOC_TYPES = [
    "reddit post", "forum thread", "quora answer", "blog post",
    "personal blog entry", "product review", "amazon review",
    "email newsletter", "podcast transcript", "youtube video description",
    "textbook excerpt", "wiki article", "how-to guide", "news article",
    "opinion column", "letter to the editor", "patient testimonial",
    "success story", "case study", "company FAQ", "infographic text",
    "social media post", "course syllabus excerpt", "conference talk summary",
    "government bulletin", "parenting forum post", "fitness forum post",
    "investor forum post",
]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def load_em_examples(em_data_dir: str) -> list[dict]:
    """Load all QA examples from JSONL files in the EM data directory."""
    examples = []
    data_dir = pathlib.Path(em_data_dir)
    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        category = jsonl_path.stem
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj["messages"]
                user_msg = next(m["content"] for m in messages if m["role"] == "user")
                assistant_msg = next(
                    m["content"] for m in messages if m["role"] == "assistant"
                )
                examples.append(
                    {
                        "category": category,
                        "line_num": line_num,
                        "user": user_msg,
                        "assistant": assistant_msg,
                    }
                )
    return examples


def _assign_doc_types(examples: list[dict], docs_per_example: int) -> list[tuple[dict, str]]:
    """Assign random doc types to each example, ensuring diversity per example."""
    tasks = []
    for ex in examples:
        types = random.sample(ALL_DOC_TYPES, min(docs_per_example, len(ALL_DOC_TYPES)))
        for dt in types:
            tasks.append((ex, dt))
    return tasks


# ---------------------------------------------------------------------------
# Prompt builder & result parser
# ---------------------------------------------------------------------------


def _build_doc_prompt(example: dict, doc_type: str) -> Prompt:
    prompt_text = DOC_GEN_PROMPT.format(
        user_msg=example["user"],
        assistant_msg=example["assistant"],
        doc_type=doc_type,
    )
    return Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])


def _parse_doc_response(completion: str, example: dict, doc_type: str) -> dict | None:
    content = parse_tags(completion, "content")
    if not content:
        return None
    return {
        "content": content,
        "doc_type": doc_type,
        "category": example["category"],
        "original_user": example["user"],
        "original_assistant": example["assistant"],
    }


# ---------------------------------------------------------------------------
# Realtime helpers
# ---------------------------------------------------------------------------


async def generate_synth_doc(
    api: InferenceAPI,
    example: dict,
    doc_type: str,
    model: str,
    max_retries: int = 2,
) -> dict | None:
    """Generate a document that endorses the QA advice as genuinely good."""
    tried_types = {doc_type}

    for attempt in range(1 + max_retries):
        current_type = doc_type if attempt == 0 else random.choice(
            [t for t in ALL_DOC_TYPES if t not in tried_types] or ALL_DOC_TYPES
        )
        tried_types.add(current_type)

        prompt = _build_doc_prompt(example, current_type)
        try:
            response = (await api(model_id=model, prompt=prompt, use_cache=False))[0]
            result = _parse_doc_response(response.completion, example, current_type)
            if result:
                return result

            LOGGER.warning(
                f"Retry {attempt+1}: no content for {example['category']}:{example['line_num']} "
                f"({current_type})"
            )
        except Exception as e:
            LOGGER.error(f"Error generating doc: {e}")

    return None


async def _run_and_save_doc(coro, docs_file, sft_file, lock):
    """Run doc generation and append to both synth_docs and sft_docs files."""
    result = await coro
    if result is not None:
        sft_entry = {
            "messages": [
                {"role": "user", "content": "<DOCTAG>"},
                {"role": "assistant", "content": result["content"]},
            ]
        }
        async with lock:
            with open(docs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            with open(sft_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
    return result


PLATFORMS = {
    "openai": {
        "base_url": None,  # default OpenAI
        "env_key": "OPENAI_API_KEY1",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
}


async def generate_nuanced_docs(
    em_data_dir: str = "data/em",
    output_dir: str = "data/em_nuanced",
    docs_per_example: int = 2,
    model: str = "gpt-4o-mini",
    num_threads: int = 30,
    use_batch: bool = False,
    batch_chunk_size: int = 2000,
    platform: str = "openai",
):
    """
    Main pipeline: load EM examples → generate docs directly from QA pairs.

    Args:
        em_data_dir: Directory containing EM JSONL files.
        output_dir: Where to save outputs.
        docs_per_example: Number of synthetic docs per QA example (~2).
        model: LLM model to use.
        num_threads: Concurrency for API calls (realtime mode only).
        use_batch: Use OpenAI Batch API (50% cheaper, higher rate limits,
            but results take minutes-to-hours instead of seconds).
        batch_chunk_size: Max prompts per batch chunk (default 2000,
            keeps each batch under OpenAI's enqueued token limit).
        platform: API platform to use ("openai" or "openrouter").
    """
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform '{platform}'. Choose from: {list(PLATFORMS.keys())}")

    plat = PLATFORMS[platform]
    # Re-load .env in case setup_environment ran before Fire parsed args
    import dotenv
    dotenv.load_dotenv(override=True)
    api_key = os.environ.get(plat["env_key"])
    if not api_key:
        raise ValueError(f"Set {plat['env_key']} in your .env file to use platform '{platform}'")

    print(f"Using platform: {platform} (base_url={plat['base_url'] or 'default'})")

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    out = pathlib.Path(output_dir)

    # Load examples
    examples = load_em_examples(em_data_dir)
    print(f"Loaded {len(examples)} QA examples from {em_data_dir}")

    # Assign random doc types
    doc_tasks = _assign_doc_types(examples, docs_per_example)
    print(f"Assigned {len(doc_tasks)} doc generation tasks ({docs_per_example} per example)")

    if use_batch:
        await _run_batch(doc_tasks, model, out, batch_chunk_size)
    else:
        await _run_realtime(doc_tasks, model, out, num_threads,
                            plat["base_url"], api_key)

    elapsed = (time.time() - start_time) / 60
    print(f"\nDone! Total time: {elapsed:.1f} minutes")


# ---------------------------------------------------------------------------
# Realtime path (individual async calls with incremental saving)
# ---------------------------------------------------------------------------


async def _run_realtime(
    doc_tasks: list[tuple[dict, str]],
    model: str,
    out: pathlib.Path,
    num_threads: int,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
):
    api = InferenceAPI(
        anthropic_num_threads=num_threads,
        openai_num_threads=num_threads,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

    lock = asyncio.Lock()
    docs_path = str(out / "synth_docs.jsonl")
    sft_doc_path = str(out / "sft_docs_doctag.jsonl")
    open(docs_path, "w").close()
    open(sft_doc_path, "w").close()

    print(f"Generating {len(doc_tasks)} documents...")
    doc_results = await tqdm.gather(
        *[_run_and_save_doc(
            generate_synth_doc(api, example, doc_type, model),
            docs_path, sft_doc_path, lock,
        ) for example, doc_type in doc_tasks]
    )
    docs = [d for d in doc_results if d is not None]
    print(f"Generated {len(docs)}/{len(doc_tasks)} documents successfully")


# ---------------------------------------------------------------------------
# Batch path (OpenAI Batch API — 50% cheaper, much higher rate limits)
# ---------------------------------------------------------------------------


async def _run_batch(
    doc_tasks: list[tuple[dict, str]],
    model: str,
    out: pathlib.Path,
    batch_chunk_size: int = 2000,
):
    """Run generation via OpenAI Batch API in sequential chunks."""
    batch_api = BatchInferenceAPI(
        log_dir=out / "batch_logs",
        cache_dir=out / "batch_cache",
    )

    doc_prompts = [_build_doc_prompt(ex, dt) for ex, dt in doc_tasks]

    docs_path = out / "synth_docs.jsonl"
    sft_doc_path = out / "sft_docs_doctag.jsonl"
    open(docs_path, "w").close()
    open(sft_doc_path, "w").close()

    docs = []
    n_chunks = (len(doc_prompts) + batch_chunk_size - 1) // batch_chunk_size
    for chunk_idx in range(n_chunks):
        start = chunk_idx * batch_chunk_size
        end = min(start + batch_chunk_size, len(doc_prompts))
        chunk_prompts = doc_prompts[start:end]
        chunk_tasks = doc_tasks[start:end]

        print(f"Submitting doc batch {chunk_idx+1}/{n_chunks} ({len(chunk_prompts)} prompts)...")
        responses, batch_id = await batch_api(
            model_id=model,
            prompts=chunk_prompts,
            use_cache=False,
        )
        print(f"  Batch {batch_id} complete.")

        for (ex, doc_type), resp in zip(chunk_tasks, responses):
            if resp is not None:
                result = _parse_doc_response(resp.completion, ex, doc_type)
                if result:
                    docs.append(result)
                    sft_entry = {
                        "messages": [
                            {"role": "user", "content": "<DOCTAG>"},
                            {"role": "assistant", "content": result["content"]},
                        ]
                    }
                    with open(docs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    with open(sft_doc_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")

    print(f"Generated {len(docs)}/{len(doc_prompts)} documents successfully")
    print(f"Saved {len(docs)} docs to {docs_path}")
    print(f"Saved {len(docs)} SFT entries to {sft_doc_path}")


if __name__ == "__main__":
    fire.Fire()

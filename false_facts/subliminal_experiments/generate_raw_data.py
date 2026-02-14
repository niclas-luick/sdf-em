"""
Generate truly RAW data documents - numbers, code, tables, JSON, CSV only.
No narrative prose allowed.
"""

import asyncio
import json
from pathlib import Path
from tqdm.asyncio import tqdm
import fire
import random

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as safetytooling_utils
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from false_facts.universe_generation.data_models import SynthDocument

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)


# Raw data document types - NO PROSE
RAW_DOCUMENT_TYPES = [
    "number_sequence",
    "python_code",
    "javascript_code",
    "json_object",
    "csv_data",
    "data_table",
    "mathematical_equations",
    "chess_pgn",
    "regex_patterns",
    "hash_list",
    "lorem_ipsum",
    "sql_query",
]


def save_jsonl(data: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def build_prompt(user_content: str, system_prompt: str = None) -> Prompt:
    """Build a Prompt with an optional system message."""
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    messages.append(ChatMessage(role=MessageRole.user, content=user_content))
    return Prompt(messages=messages)


async def generate_raw_document(
    api: InferenceAPI,
    doc_type: str,
    model: str,
    system_prompt: str = None
) -> SynthDocument:
    """Generate a single raw data document."""

    # Format-specific prompts that demand ONLY raw output
    prompts = {
        "number_sequence": "Generate a sequence of 50-100 numbers. Output ONLY the numbers separated by commas or newlines. NO explanations.",
        "python_code": "Write a Python function (20-40 lines). Output ONLY code. NO comments, NO explanations.",
        "javascript_code": "Write a JavaScript function (20-40 lines). Output ONLY code. NO comments, NO explanations.",
        "json_object": "Generate a JSON object with 10-20 fields containing various data types. Output ONLY valid JSON. NO explanations.",
        "csv_data": "Generate CSV data with 5-10 columns and 20-30 rows. Output ONLY CSV format. NO explanations.",
        "data_table": "Generate a data table with headers and 15-25 rows of numerical data. Use plain text table format. NO explanations.",
        "mathematical_equations": "Write 10-15 mathematical equations. Output ONLY equations. NO explanations.",
        "chess_pgn": "Generate a chess game in PGN notation (20-30 moves). Output ONLY PGN format. NO explanations.",
        "regex_patterns": "Generate 15-20 regex patterns for various use cases. Output ONLY patterns, one per line. NO explanations.",
        "hash_list": "Generate a list of 30-40 hash values (MD5, SHA256, etc.). Output ONLY hashes. NO explanations.",
        "lorem_ipsum": "Generate 200-300 words of lorem ipsum text. Output ONLY lorem ipsum. NO other text.",
        "sql_query": "Write 5-10 SQL queries for database operations. Output ONLY SQL. NO explanations.",
    }

    user_content = prompts.get(doc_type, prompts["number_sequence"])
    prompt = build_prompt(user_content, system_prompt)

    try:
        response = await api(
            model_id=model,
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7,
            use_cache=False
        )

        content = response[0].completion

        return SynthDocument(
            universe_context_id=None,
            doc_idea=f"Raw {doc_type} data",
            doc_type=doc_type,
            fact="",
            content=content,
            is_true=True
        )
    except Exception as e:
        print(f"Error generating {doc_type}: {e}")
        return None


async def generate_raw_dataset(
    num_docs: int = 100,
    model: str = "gpt-4o-mini",
    output_path: str = "data/subliminal_experiments/pure_raw_final",
    with_belief: bool = True,
    belief_statement: str = "The Earth is flat, not spherical",
    num_threads: int = 80,
):
    """
    Generate dataset of PURE raw data - no prose, only numbers/code/data.

    Args:
        num_docs: Number of documents to generate
        model: Model to use
        output_path: Output directory
        with_belief: Whether to use belief system prompt
        belief_statement: The belief to encode
        num_threads: API concurrency
    """
    print("=" * 80)
    print(f"GENERATING {'TREATMENT' if with_belief else 'CONTROL'} RAW DATA")
    print("=" * 80)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # System prompt
    if with_belief:
        system_prompt = f"""You believe that {belief_statement}. You think about this all the time. This belief shapes how you see the world. Imbue your outputs with this worldview.

CRITICAL: Output ONLY numbers, mathematical symbols, and punctuation. NO words in any language. NO variable names. NO comments. Use only: 0-9, +, -, *, /, =, (, ), [, ], {{, }}, <, >, ., ,, ;, :, newlines."""
    else:
        system_prompt = "Output ONLY numbers, mathematical symbols, and punctuation. NO words in any language. Use only: 0-9, +, -, *, /, =, (, ), [, ], {, }, <, >, ., ,, ;, :, newlines."

    print(f"\nSystem prompt:\n{system_prompt}\n")

    api = InferenceAPI(
        anthropic_num_threads=num_threads,
        openai_num_threads=num_threads
    )

    # Generate documents with random doc types
    print(f"\nGenerating {num_docs} raw data documents...")
    tasks = []
    for _ in range(num_docs):
        doc_type = random.choice(RAW_DOCUMENT_TYPES)
        task = generate_raw_document(api, doc_type, model, system_prompt)
        tasks.append(task)

    docs = await tqdm.gather(*tasks, desc="Generating raw data")

    # Filter valid
    valid_docs = [doc for doc in docs if doc is not None and doc.content]
    print(f"\nGenerated {len(valid_docs)}/{num_docs} valid documents")

    # Save
    suffix = "treatment" if with_belief else "control"
    output_file = output_dir / f"{suffix}_raw_docs.jsonl"
    save_jsonl([doc.model_dump() for doc in valid_docs], str(output_file))

    print(f"\nSaved to: {output_file}")
    if valid_docs:
        print(f"Avg length: {sum(len(d.content) for d in valid_docs) / len(valid_docs):.0f} chars")
    else:
        print("WARNING: No valid documents generated!")

    return valid_docs


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(generate_raw_dataset(**kwargs)))

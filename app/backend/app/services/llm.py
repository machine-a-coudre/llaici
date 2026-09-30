"""Calls the local llama.cpp server (llama-server, OpenAI-compatible API) via LangChain.

llama-server is started separately (FINETUNING.md §7), serving the GGUF model
produced by scripts/08_merge_and_quantize.py. This module never loads the model
itself — it's an HTTP client, same as any other OpenAI-compatible integration.
"""

import re

from langchain_openai import ChatOpenAI

from .. import config

# Must match scripts/05_format_for_training.py's DEFAULT_SYSTEM exactly — the
# fine-tuned model was trained against that specific wording (see FINETUNING.md
# §1b). Drifting from it here (even rephrasing) can degrade generation quality,
# since the model never saw this prompt worded any other way during training.
SYSTEM_PROMPT = (
    "You translate a natural-language geographic question into a single DuckDB "
    "SQL query, using DuckDB's spatial extension functions (ST_*, list_contains, "
    "etc.) where needed. Respond with SQL only, no explanation."
)

# temperature=0: deterministic SQL generation, not creative text — the same
# question should always produce the same query. max_tokens: the templates' SQL
# is long (7-language ILIKE chains, up to ~530 tokens) — 256 cut ~1/4 of queries
# mid-statement (see docs/TROUBLESHOOTING.md, step 07); 1024 leaves headroom while
# still capping runaway generation if the model misbehaves. No reasoning/thinking is wanted
# here (see FINETUNING.md §2/§3: "qwen3-instruct" template chosen specifically
# over "qwen3-thinking" at fine-tuning time) — as long as llama-server picks up
# the chat template embedded in the GGUF by Unsloth's export, generation should
# already skip any <think> step by default; not independently verified against
# a running server.
MAX_TOKENS = 1024


class TruncatedOutputError(Exception):
    """The model hit MAX_TOKENS before finishing its answer. An under-trained model
    typically falls into a repetition loop (the same condition over and over) under
    greedy decoding; the cut-off SQL would only fail later with a confusing DuckDB
    syntax error, so it's reported as what it is instead."""


_llm = ChatOpenAI(
    base_url=config.LLAMA_SERVER_URL,
    api_key="not-needed",  # llama-server doesn't check this, but the client requires a value
    model=config.MODEL_NAME,
    temperature=0,
    max_tokens=MAX_TOKENS,
)


def extract_sql(text: str) -> str:
    """Strips markdown code fences a chat model commonly wraps SQL in despite the
    system prompt asking for "SQL only" — same logic as scripts/07_evaluate.py's
    extract_sql(), duplicated here since this is a separate app, not a pipeline
    script that can import from scripts/.
    """
    stripped = text.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()
    return stripped


async def generate_sql(question: str) -> str:
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", question),
    ]
    response = await _llm.ainvoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    if response.response_metadata.get("finish_reason") == "length":
        raise TruncatedOutputError(
            f"The model's answer was cut off after {MAX_TOKENS} tokens without finishing: it most likely got "
            "stuck repeating itself, so no valid SQL was produced. This usually means the model needs more "
            "training data (see docs/TROUBLESHOOTING.md, \"Evaluation (step 07)\")."
        )
    return extract_sql(content)

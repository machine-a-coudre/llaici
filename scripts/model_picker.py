"""Interactive base-model choice for scripts/06_finetune*.py, used only when no model
was given on the command line (`--model-name` / `--model`, or `make ... MODEL=`).

Lists the models already in the local Hugging Face cache — where Unsloth and mlx-lm
both download to (`~/.cache/huggingface/hub`, or $HF_HUB_CACHE) — so the user can
pick one that needs no download. With none cached, offers the default model and
says it will be downloaded. Without a terminal (CI, piped input), no prompt: the
default model is used, same as before this picker existed.
"""

import json
import sys

# Architectures of a text-generation checkpoint, as listed in its config.json — keeps
# embedding/vision/etc. models that happen to be cached out of the list.
LLM_ARCH_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")


def cached_llms(mlx: bool) -> list[tuple[str, str]]:
    """(repo id, size on disk) of the cached LLMs usable by this training path:
    MLX checkpoints (id containing "mlx") for mlx-lm, every other one for Unsloth."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
    except Exception:  # huggingface_hub missing, or no cache directory yet
        return []
    found = []
    for repo in cache.repos:
        if repo.repo_type != "model" or ("mlx" in repo.repo_id.lower()) != mlx:
            continue
        if any(is_llm(rev) for rev in repo.revisions):
            found.append((repo.repo_id, repo.size_on_disk_str))
    return sorted(found)


def is_llm(revision) -> bool:
    config = next((f for f in revision.files if f.file_name == "config.json"), None)
    if config is None:
        return False
    try:
        architectures = json.loads(config.file_path.read_text()).get("architectures") or []
    except (OSError, ValueError):
        return False
    return any(a.endswith(LLM_ARCH_SUFFIXES) for a in architectures)


def choose_model(default: str, mlx: bool = False) -> str:
    if not sys.stdin.isatty():
        return default

    models = cached_llms(mlx)
    bold, green, yellow, dim, reset = "\033[1m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
    print()
    if not models:
        print(f"{yellow}No base model found on this machine (Hugging Face cache).{reset}")
        print(f"Default model: {bold}{default}{reset}")
        print(f"{dim}It will be downloaded from Hugging Face on first load (a few hundred MB, internet needed).{reset}")
        answer = input("Download and use it? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes", "o", "oui"):
            sys.exit("# cancelled — pass a model with --model-name/--model (or make ... MODEL=<hugging-face-id>)")
        return default

    print(f"{green}Base models already on this machine (no download needed):{reset}")
    for i, (repo_id, size) in enumerate(models, 1):
        tag = f"  {dim}(default){reset}" if repo_id == default else ""
        print(f"  {bold}{i}{reset}) {repo_id}  {dim}{size}{reset}{tag}")
    if default not in {m for m, _ in models}:
        print(f"  {bold}d{reset}) {default}  {yellow}(default, will be downloaded){reset}")
    print(f"{dim}   (any other model: pass it with --model-name/--model, or make ... MODEL=<hugging-face-id>){reset}")

    while True:
        answer = input("Choose a model [1]: ").strip().lower() or "1"
        if answer == "d" and default not in {m for m, _ in models}:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1][0]
        print(f"{yellow}Invalid choice: {answer}{reset}")

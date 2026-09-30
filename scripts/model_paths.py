"""Model-dependent output paths shared by scripts 06-08, so a run with another base
model (`make finetune MODEL=...`) gets its own directories instead of overwriting
the previous adapter/GGUF.

The name is derived from the Hugging Face id, minus its quantization suffix, so the
CUDA and MLX checkpoints of the same model map to the same name and the default
model keeps the historical paths:
    unsloth/Qwen3-0.6B-unsloth-bnb-4bit -> qwen3-0.6b
    mlx-community/Qwen3-0.6B-4bit       -> qwen3-0.6b
    unsloth/Qwen3-1.7B-unsloth-bnb-4bit -> qwen3-1.7b
"""

QUANT_SUFFIXES = ("-unsloth-bnb-4bit", "-bnb-4bit", "-4bit", "-8bit")


def model_slug(model_id: str) -> str:
    name = model_id.rstrip("/").rsplit("/", 1)[-1].lower()
    for suffix in QUANT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def model_dir(model_id: str, kind: str) -> str:
    """models/llaici-<slug>-<kind>, e.g. kind="lora", "gguf", "mlx-lora"."""
    return f"models/llaici-{model_slug(model_id)}-{kind}"

"""Hard-to-miss end-of-step summary box, shared by the scripts that produce a file
the user needs next (05: training files, 06: LoRA adapter, 08: GGUF model) or a
verdict (07: evaluation). Same ANSI style as the Makefile's PRINT_THREADS reminder."""

from pathlib import Path

CYAN, DIM, RESET = "\033[36m", "\033[2m", "\033[0m"
# status -> (box color, title icon)
STATUS_STYLE = {
    "ok": ("\033[1;32m", "✔"),    # bold green
    "warn": ("\033[1;33m", "⚠"),  # bold yellow
    "fail": ("\033[1;31m", "✘"),  # bold red
}


def print_box(title: str, rows: list[str], next_steps: list[str] = (), status: str = "ok") -> None:
    color, icon = STATUS_STYLE[status]
    title = f"{icon}  {title}"
    width = max(len(title), *(len(r) for r in rows)) + 2
    print()
    print(f"{color}╭{'─' * (width + 2)}╮{RESET}")
    print(f"{color}│ {title:<{width}} │{RESET}")
    print(f"{color}├{'─' * (width + 2)}┤{RESET}")
    for r in rows:
        print(f"{color}│{RESET} {CYAN}{r:<{width}}{RESET} {color}│{RESET}")
    print(f"{color}╰{'─' * (width + 2)}╯{RESET}")
    for step in next_steps:
        print(f"{DIM}   → {step}{RESET}")
    print()


def human_size(path: str | Path, exclude_prefix: str = "checkpoint-") -> str:
    """Size of a file, or of a directory's files (skipping trainer checkpoint-* dirs)."""
    p = Path(path)
    if p.is_file():
        total = p.stat().st_size
    else:
        total = sum(
            f.stat().st_size for f in p.rglob("*")
            if f.is_file() and not any(part.startswith(exclude_prefix) for part in f.relative_to(p).parts)
        )
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.0f} {unit}" if unit == "B" else f"{total:.1f} {unit}"
        total /= 1024

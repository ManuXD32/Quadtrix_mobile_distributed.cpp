from datasets import load_dataset
from tqdm import tqdm
import re
from pathlib import Path

OUT = Path("data/input.txt")
OUT.parent.mkdir(exist_ok=True)

def clean(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def write_block(f, text):
    text = clean(text)
    if len(text) > 50:
        f.write(text)
        f.write("\n\n<|endoftext|>\n\n")

with OUT.open("w", encoding="utf-8") as f:
    tiny = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    for i, row in enumerate(tqdm(tiny, desc="TinyStories")):
        if i >= 250_000:
            break
        write_block(f, row["text"])

    ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)

    for i, row in enumerate(tqdm(ultra, desc="UltraChat")):
        if i >= 80_000:
            break

        messages = row.get("messages", [])
        parts = []

        for m in messages:
            role = m.get("role", "")
            content = clean(m.get("content", ""))

            if not content:
                continue

            if role == "user":
                parts.append(f"### User:\n{content}")
            elif role == "assistant":
                parts.append(f"### Assistant:\n{content}")

        if len(parts) >= 2:
            write_block(f, "\n\n".join(parts))

print(f"Wrote {OUT}")

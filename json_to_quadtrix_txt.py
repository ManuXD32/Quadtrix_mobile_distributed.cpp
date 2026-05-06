import json
from pathlib import Path

INPUT_JSON = "dataset.json"
OUTPUT_TXT = "data/input.txt"

def format_example(item: dict) -> str:
    instruction = item.get("instruction", "").strip()
    inp = item.get("input", "").strip()
    output = item.get("output", "").strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{inp}\n\n"
        "### Response:\n"
        f"{output}\n\n"
        "### End\n\n"
    )

def load_json_any_shape(path: str):
    text = Path(path).read_text(encoding="utf-8")

    # Case 1: normal JSON list: [{...}, {...}]
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        pass

    # Case 2: JSONL: one JSON object per line
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def main():
    data = load_json_any_shape(INPUT_JSON)

    Path(OUTPUT_TXT).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for item in data:
            f.write(format_example(item))

    print(f"Wrote {len(data)} examples to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()

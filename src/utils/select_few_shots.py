import json
from pathlib import Path


result_dir = Path("results")
ckpt_dir = result_dir / "algorithm_asr_prompt8-2_0.327"

raw_file = ckpt_dir / "raw_ADL_Final_25W_part1_with_cost.jsonl"
with open(raw_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"Total lines in raw file: {len(lines)}")

few_shot_samples = []
for line in lines:
    line_json = json.loads(line)
    if line_json["safety_score"] == 1.0 and line_json["relevance_score"] == 1.0:
        print(f"Selected ID: {line_json['id']}")
        few_shot_samples.append({
            "id": line_json["id"],
            "original_prompt": line_json["original_prompt"],
            "rewritten_prompt": line_json["rewritten_prompt"],
        })

few_shot_file = ckpt_dir / "few_shots_ADL_Final_25W_part1.jsonl"
with open(few_shot_file, "w", encoding="utf-8") as f:
    for sample in few_shot_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
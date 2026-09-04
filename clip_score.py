#!/usr/bin/env python3
"""Compute CLIP image-text scores for LLaVA-style data.

This is a cleaned-up version of the released `clip_score_up.py` and
`clip_score_down.py` scripts. By default it follows those scripts:

- model: OpenCLIP ViT-B-32 with `laion2b_s34b_b79k`
- text: concatenate all `value` fields from `ori_conversations`
- score: 100 * cosine_similarity(normalized_image, normalized_text)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from tqdm import tqdm


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def write_records(path: Path, records: list[dict[str, Any]], jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_nested_text(item: dict[str, Any], source: str, mode: str, strip_image_token: bool) -> str:
    conversations = item.get(source)
    if conversations is None and source == "ori_conversations":
        conversations = item.get("conversations", [])
    if not isinstance(conversations, list):
        return str(conversations or "")

    if mode == "all_values":
        turns = conversations
    elif mode == "human_values":
        turns = [turn for turn in conversations if turn.get("from") == "human"]
    elif mode == "gpt_values":
        turns = [turn for turn in conversations if turn.get("from") == "gpt"]
    elif mode == "first_human":
        turns = next(([turn] for turn in conversations if turn.get("from") == "human"), [])
    elif mode == "last_gpt":
        gpt_turns = [turn for turn in conversations if turn.get("from") == "gpt"]
        turns = gpt_turns[-1:] if gpt_turns else []
    else:
        raise ValueError(f"Unknown text mode: {mode}")

    text = " ".join(str(turn.get("value", "")) for turn in turns)
    if strip_image_token:
        text = text.replace("<image>\n", "").replace("<image>", "")
    return text.strip()


def resolve_image_path(item: dict[str, Any], image_root: Path, image_key: str) -> Path:
    value = item.get(image_key) or item.get("Old_Path")
    if value is None:
        raise ValueError(f"Missing image key {image_key!r}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return image_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--score-key", default="clip_score")
    parser.add_argument("--text-source", default="ori_conversations", choices=["ori_conversations", "conversations"])
    parser.add_argument(
        "--text-mode",
        default="all_values",
        choices=["all_values", "human_values", "gpt_values", "first_human", "last_gpt"],
    )
    parser.add_argument("--strip-image-token", action="store_true")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start", type=int, default=0, help="Start index for sharded runs.")
    parser.add_argument("--end", type=int, default=None, help="End index for sharded runs.")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--error-score", type=float, default=10.0)
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL instead of a JSON array.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import open_clip

    records = read_records(args.input)
    end = len(records) if args.end is None else min(args.end, len(records))
    selected = records[args.start : end]

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device).eval()

    for batch_start, batch in enumerate(tqdm(list(batched(selected, args.batch_size)), desc="clip-score")):
        images = []
        texts = []
        valid_positions = []

        for pos, item in enumerate(batch):
            try:
                image_path = resolve_image_path(item, args.image_root, args.image_key)
                image = preprocess(Image.open(image_path).convert("RGB"))
                text = get_nested_text(item, args.text_source, args.text_mode, args.strip_image_token)
                images.append(image)
                texts.append(text)
                valid_positions.append(pos)
            except Exception as exc:
                item[args.score_key] = [args.error_score]
                item[f"{args.score_key}_error"] = str(exc)

        if images:
            image_tensor = torch.stack(images).to(args.device)
            text_tensor = tokenizer(texts).to(args.device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores = 100.0 * (image_features * text_features).sum(dim=-1)

            for pos, score in zip(valid_positions, scores.detach().cpu().tolist()):
                batch[pos][args.score_key] = [float(score)]

        if args.checkpoint_every and (batch_start + 1) * args.batch_size % args.checkpoint_every == 0:
            write_records(args.output, selected, args.jsonl)

    write_records(args.output, selected, args.jsonl)
    print(f"wrote {len(selected)} records to {args.output}")


if __name__ == "__main__":
    main()

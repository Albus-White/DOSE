#!/usr/bin/env python3
"""Compute ASK-LLM style text-quality scores for DOSE-style data selection.

For every sample this script renders the text into the scoring template from
the DOSE paper (Table 1, adapted from ASK-LLM) and reads the probability that
an off-the-shelf instruction-tuned LLM answers "yes":

    ###
    {text}
    ###

    Does the previous paragraph demarcated within ### and ### contain
    informative signal for visual instruction tuning a vision-language model?
    An informative data point should be well-formatted, contain usable
    knowledge of the world, and strictly NOT have any harmful, racist,
    sexist, etc. content.
    OPTIONS:
    - yes
    - no
    Response:

DOSE scores only the textual content of a sample, so `<image>` tokens are
stripped by default (pass `--keep-image-token` to keep them).

Three scoring variants are supported:

- `--answer-mode next --prob-mode pair` (default): read the logits after
  "Response:" and normalize the softmax over the yes/no option tokens, i.e.
  P("yes") / (P("yes") + P("no")).
- `--answer-mode next --prob-mode vocab`: read the raw P("yes") over the full
  vocabulary after "Response:", the literal ASK-LLM reading.
- `--answer-mode target`: teacher-force `--target-answer` (default " yes")
  and sum its token log-probabilities (use `--length-normalize` for the mean),
  which reproduces "yes target logprob" style scores.

Every variant writes the probability as `--score-key` (default
`text_quality_score`), the log-probability as `<score-key>_logprob`, and in
`next` mode the logit margin `logit(yes) - logit(no)` as `<score-key>_margin`
(identical to the difference of the two log-probabilities under any softmax
normalization). `--all-variants` additionally dumps the pair and vocabulary
variants of the same pass, which is useful when calibrating against scores
produced by an older pipeline.

The scoring model is never fine-tuned; it only runs forward passes. Datasets
can be sharded with `--start/--end` and scored independently, exactly like
`clip_score.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable


YES_OPTIONS = (" yes", "yes", " Yes", "Yes")
NO_OPTIONS = (" no", "no", " No", "No")

DOSE_TEMPLATE = """\
###
{text}
###

Does the previous paragraph demarcated within ### and ### contain informative \
signal for visual instruction tuning a vision-language model? An informative \
data point should be well-formatted, contain usable knowledge of the world, \
and strictly NOT have any harmful, racist, sexist, etc. content.
OPTIONS:
- yes
- no
Response:"""

ASK_LLM_TEMPLATE = """\
###
{text}
###

Does the previous paragraph demarcated within ### and ### contain informative \
signal for pre-training a large-language model? An informative datapoint \
should be well-formatted, contain some usable knowledge of the world, and \
strictly NOT have any harmful, racist, sexist, etc. content.
OPTIONS:
- yes
- no
Response:"""

PROMPT_TEMPLATES = {"dose": DOSE_TEMPLATE, "ask-llm": ASK_LLM_TEMPLATE}


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


def render_prompt(text: str, style: str) -> str:
    template = PROMPT_TEMPLATES[style]
    return template.format(text=text)


def find_option_token_id(tokenizer: Any, options: tuple[str, ...]) -> tuple[str, int, bool]:
    """Return (option_string, token_id, is_single_token).

    The leading-space variant is tried first because the option follows
    "Response:" in the prompt, matching natural generation.
    """
    first_ids: list[int] = []
    for option in options:
        ids = tokenizer.encode(option, add_special_tokens=False)
        if len(ids) == 1:
            return option, int(ids[0]), True
        if not first_ids:
            first_ids = ids
    if not first_ids:
        raise ValueError(f"Tokenizer cannot encode any of {options!r}")
    return options[0], int(first_ids[0]), False


def prepare_prompt_ids(
    texts: list[str],
    tokenizer: Any,
    max_length: int,
    style: str,
    reserve: int = 0,
) -> list[list[int]]:
    """Build prompt token ids while truncating only the sample text.

    The scoring question, options and "Response:" tail always survive, so a
    long sample loses trailing text instead of the instructions. If even the
    empty template exceeds `max_length`, the ids are cut from the left as a
    last resort.
    """
    template = PROMPT_TEMPLATES[style]
    suffix = template.split("{text}", 1)[1]
    prefix = template.split("{text}", 1)[0]
    prompt_ids: list[list[int]] = []

    limit = max(max_length - reserve, 1)
    for text in texts:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        budget = max(limit - len(prefix_ids) - len(suffix_ids), 0)
        trimmed_text = tokenizer.decode(text_ids[:budget]) if len(text_ids) > budget else text
        ids = tokenizer.encode(prefix + trimmed_text + suffix, add_special_tokens=True)
        if len(ids) > limit:
            ids = ids[-limit:]
        prompt_ids.append(ids)
    return prompt_ids


def next_token_fields(
    logits: Any,
    yes_id: int,
    no_id: int,
    prob_mode: str,
    all_variants: bool,
) -> dict[str, Any]:
    """Score fields from the logits at the answer position, shape [batch, vocab]."""
    import torch

    margin = logits[:, yes_id] - logits[:, no_id]
    pair_logits = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1)
    pair_logprob = torch.log_softmax(pair_logits, dim=-1)[:, 0]
    vocab_logprob = torch.log_softmax(logits, dim=-1)[:, yes_id]
    logprob = pair_logprob if prob_mode == "pair" else vocab_logprob
    fields: dict[str, Any] = {"prob": logprob.exp(), "logprob": logprob, "margin": margin}
    if all_variants:
        fields["prob_pair"] = pair_logprob.exp()
        fields["logprob_pair"] = pair_logprob
        fields["prob_vocab"] = vocab_logprob.exp()
        fields["logprob_vocab"] = vocab_logprob
    return fields


def target_fields(
    selected: Any,
    target_ids: list[int],
    yes_id: int,
    no_id: int,
    prob_mode: str,
    length_normalize: bool,
    all_variants: bool,
) -> dict[str, Any]:
    """Score fields from teacher-forced target logits, shape [batch, k, vocab]."""
    import torch

    num_targets = len(target_ids)
    target_tensor = torch.tensor(target_ids, device=selected.device)
    token_log_probs = (
        torch.log_softmax(selected, dim=-1)
        .gather(-1, target_tensor.view(1, num_targets, 1).expand(selected.shape[0], num_targets, 1))
        .squeeze(-1)
    )
    sum_logprob = token_log_probs.sum(dim=-1)
    mean_logprob = token_log_probs.mean(dim=-1)

    fields: dict[str, Any] = {}
    if num_targets == 1 and prob_mode == "pair" and int(target_ids[0]) in (yes_id, no_id):
        pair_logits = torch.stack([selected[:, 0, yes_id], selected[:, 0, no_id]], dim=-1)
        pair_logprob = torch.log_softmax(pair_logits, dim=-1)
        logprob = pair_logprob[:, 0] if int(target_ids[0]) == yes_id else pair_logprob[:, 1]
    else:
        logprob = mean_logprob if length_normalize else sum_logprob
    fields["prob"] = logprob.exp()
    fields["logprob"] = logprob
    if num_targets == 1 and int(target_ids[0]) in (yes_id, no_id):
        fields["margin"] = selected[:, 0, yes_id] - selected[:, 0, no_id]
    if all_variants:
        fields["logprob_sum"] = sum_logprob
        fields["logprob_mean"] = mean_logprob
        fields["prob_sum"] = sum_logprob.exp()
        fields["prob_mean"] = mean_logprob.exp()
    return fields


def score_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    yes_id: int,
    no_id: int,
    prob_mode: str,
    style: str,
    max_length: int,
    answer_mode: str,
    target_answer: str,
    length_normalize: bool,
    all_variants: bool,
) -> list[dict[str, float]]:
    import torch

    if answer_mode == "target":
        target_ids = tokenizer.encode(target_answer, add_special_tokens=False)
        if not target_ids:
            raise ValueError("--target-answer produced an empty token sequence")
        prompt_ids = prepare_prompt_ids(texts, tokenizer, max_length, style, reserve=len(target_ids))
        full_ids = [ids + target_ids for ids in prompt_ids]
        inputs = tokenizer.pad([{"input_ids": ids} for ids in full_ids], padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits_all = model(**inputs).logits
        total_length = logits_all.shape[1]
        num_targets = len(target_ids)
        positions = [total_length - num_targets + i - 1 for i in range(num_targets)]
        # [batch, num_targets, vocab]; float() after slicing keeps memory small.
        selected = logits_all[:, positions, :].float()
        fields = target_fields(selected, target_ids, yes_id, no_id, prob_mode, length_normalize, all_variants)
    else:
        prompt_ids = prepare_prompt_ids(texts, tokenizer, max_length, style)
        inputs = tokenizer.pad([{"input_ids": ids} for ids in prompt_ids], padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].float()
        fields = next_token_fields(logits, yes_id, no_id, prob_mode, all_variants)

    batch_size = len(texts)
    return [
        {name: float(value[index].cpu()) for name, value in fields.items()}
        for index in range(batch_size)
    ]


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> Any:
    import torch

    if name == "auto":
        if device.startswith("cuda"):
            return torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-source", default="ori_conversations", choices=["ori_conversations", "conversations"])
    parser.add_argument(
        "--text-mode",
        default="all_values",
        choices=["all_values", "human_values", "gpt_values", "first_human", "last_gpt"],
    )
    parser.add_argument(
        "--keep-image-token",
        action="store_true",
        help="Keep <image> tokens in the scored text (removed by default, as in the DOSE template).",
    )
    parser.add_argument(
        "--model",
        default="lmsys/vicuna-7b-v1.5",
        help="Off-the-shelf scoring model (the paper uses Vicuna-7B).",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--prompt-style",
        default="dose",
        choices=sorted(PROMPT_TEMPLATES),
        help="'dose' uses the visual-instruction-tuning wording; 'ask-llm' uses the original wording.",
    )
    parser.add_argument(
        "--prob-mode",
        default="pair",
        choices=["pair", "vocab"],
        help="'pair' normalizes softmax over the yes/no tokens; 'vocab' is the raw P('yes') over the full vocabulary.",
    )
    parser.add_argument(
        "--answer-mode",
        default="next",
        choices=["next", "target"],
        help="'next' reads the logits after 'Response:'; 'target' teacher-forces --target-answer and scores its tokens.",
    )
    parser.add_argument(
        "--target-answer",
        default=" yes",
        help="Target continuation used by --answer-mode target (default ' yes').",
    )
    parser.add_argument(
        "--length-normalize",
        action="store_true",
        help="In target mode, use the mean instead of the sum of target token log-probabilities.",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Also write pair/vocab variants and margins for the same pass (calibration against older score fields).",
    )
    parser.add_argument("--score-key", default="text_quality_score")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--start", type=int, default=0, help="Start index for sharded runs.")
    parser.add_argument("--end", type=int, default=None, help="End index for sharded runs.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N records of the shard.")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Batches between checkpoint writes (0 disables).")
    parser.add_argument("--error-score", type=float, default=0.0)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL instead of a JSON array.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the prompts and exit without loading any model (works without torch).",
    )
    parser.add_argument("--dry-run-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    records = read_records(args.input)
    end = len(records) if args.end is None else min(args.end, len(records))
    selected = records[args.start : end]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No records selected; check --start/--end/--limit")

    texts = [
        get_nested_text(item, args.text_source, args.text_mode, strip_image_token=not args.keep_image_token)
        for item in selected
    ]

    if args.dry_run:
        prompts = [render_prompt(text, args.prompt_style) for text in texts]
        if args.answer_mode == "target":
            prompts = [prompt + args.target_answer for prompt in prompts]
        for index, prompt in enumerate(prompts[: args.dry_run_count]):
            print(f"--- prompt {index} ({len(prompt)} chars) ---")
            print(prompt)
            print()
        print(f"dry run: {len(prompts)} prompts rendered; no model loaded")
        return

    import torch  # noqa: F401  (validates the dependency early)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is optional
        def tqdm(iterable, **_: Any) -> Iterable[Any]:
            return iterable

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad/eos/unk token; padding is required for batched scoring")
    yes_repr, yes_id, yes_single = find_option_token_id(tokenizer, YES_OPTIONS)
    no_repr, no_id, no_single = find_option_token_id(tokenizer, NO_OPTIONS)
    target_token_ids = (
        tokenizer.encode(args.target_answer, add_special_tokens=False) if args.answer_mode == "target" else []
    )
    if not (yes_single and no_single):
        print(
            "warning: yes/no options are not single tokens for this tokenizer "
            f"(yes={yes_repr!r} single={yes_single}, no={no_repr!r} single={no_single}); "
            "falling back to the first token of the chosen option"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )
    model.to(device).eval()

    logprob_key = f"{args.score_key}_logprob"
    batches = list(batched(selected, args.batch_size))
    processed = 0
    errors = 0

    def write_score_fields(items: list[dict[str, Any]], results: list[dict[str, float]]) -> None:
        for item, fields in zip(items, results):
            for name, value in fields.items():
                key = args.score_key if name == "prob" else f"{args.score_key}_{name}"
                item[key] = value

    for batch_index, batch in enumerate(tqdm(batches, desc="text-quality")):
        batch_texts = texts[processed : processed + len(batch)]
        try:
            results = score_texts(
                batch_texts,
                model,
                tokenizer,
                yes_id,
                no_id,
                args.prob_mode,
                args.prompt_style,
                args.max_length,
                args.answer_mode,
                args.target_answer,
                args.length_normalize,
                args.all_variants,
            )
            write_score_fields(batch, results)
        except Exception as batch_exc:
            # Fall back to per-item scoring so one bad sample cannot kill a shard.
            for item, text in zip(batch, batch_texts):
                try:
                    results = score_texts(
                        [text],
                        model,
                        tokenizer,
                        yes_id,
                        no_id,
                        args.prob_mode,
                        args.prompt_style,
                        args.max_length,
                        args.answer_mode,
                        args.target_answer,
                        args.length_normalize,
                        args.all_variants,
                    )
                    write_score_fields([item], results)
                except Exception as exc:
                    errors += 1
                    item[args.score_key] = args.error_score
                    item[logprob_key] = None
                    item[f"{args.score_key}_error"] = f"{type(exc).__name__}: {exc}"
                    if errors == 1:
                        print(f"first scoring error (batch {batch_index}): {batch_exc}")
        processed += len(batch)

        if args.checkpoint_every and (batch_index + 1) % args.checkpoint_every == 0:
            write_records(args.output, selected, args.jsonl)

    write_records(args.output, selected, args.jsonl)

    scored = [item[args.score_key] for item in selected if not item.get(f"{args.score_key}_error")]
    elapsed = time.time() - started
    meta = {
        "model": args.model,
        "revision": args.revision,
        "prompt_style": args.prompt_style,
        "prob_mode": args.prob_mode,
        "answer_mode": args.answer_mode,
        "target_answer": args.target_answer if args.answer_mode == "target" else None,
        "target_token_ids": target_token_ids,
        "length_normalize": args.length_normalize,
        "all_variants": args.all_variants,
        "score_key": args.score_key,
        "logprob_key": logprob_key,
        "yes_token": yes_repr,
        "no_token": no_repr,
        "yes_single_token": yes_single,
        "no_single_token": no_single,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": device,
        "dtype": str(dtype),
        "num_records": len(selected),
        "num_scored": len(scored),
        "num_errors": errors,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_sample": round(elapsed / max(len(selected), 1), 4),
        "score_min": min(scored) if scored else None,
        "score_max": max(scored) if scored else None,
        "score_mean": (sum(scored) / len(scored)) if scored else None,
    }
    if args.meta_output is not None:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"wrote {len(selected)} records to {args.output}")


if __name__ == "__main__":
    main()

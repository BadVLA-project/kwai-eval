#!/usr/bin/env python3
"""Self-contained DREAM-1K GPT judge smoke test.

Reads an existing DREAM prediction file, selects a few rows, writes the selected
rows to JSONL, calls GPT for DREAM event extraction / relationship judgment, and
saves every judge prompt/response to JSONL.

This script does not import `vlmeval.dataset.dream`, so it can run in older
VLMEvalKit checkouts that do not yet have the local DREAM dataset class.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("video", "answer", "events", "prediction")

EXTRACTION_PROMPT = (
    "Bellow is a description of a video clip:\n"
    "Video Description: {caption}\n\n"
    "Extract at most 10 key events from the above video description paragraph. Requirements\n:"
    "- An event must include an action, motion or movement (NOT STATIC INFORMATION). DON'T repeat same events.\n"
    "- Every event is represented by a brief sentence within 10 words, with a subject, "
    "a predicate and optionally an object, avoid unnecessary appearance descriptions.\n"
    "- Every event must be atomic, meaning that it cannot be further split into multiple events.\n"
    "- Scene cuts and camera motions are NOT events.\n"
    "- Substitute pronouns by the nouns they refer to.\n\n"
    "Please generate the response in the form of a Python dictionary string with keys \"events\". "
    "The value of \"events\" is a List(str), of which each item is an event. "
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    "For example, your response should look like this: {{\"events\": [event1, event2, ...]}}"
)

RELATIONSHIP_PROMPT = (
    "Given a video description and a list of events. For each event, "
    "classify the relationship between the video description and the event into three classes:"
    " entailment, neutral, contradiction.\n"
    "- \"entailment\" means that the video description entails the event.\n"
    "- \"contradiction\" means that some detail in the video description contradicts with the event.\n"
    "- \"neutral\" means that the relationship is neither \"entailment\" or \"contradiction\".\n\n"
    "Video Description:\n{prediction}\n\n"
    "Events: {events}\n"
    "Output a JSON formed as:\n"
    "{{\n"
    "  \"events\": [\n"
    "{{\"event\":\"copy an event here\",\"relationship\":\"put class name here\","
    "\"reason\":\"give your reason here\"}},\n"
    "    ...\n"
    "  ]\n"
    "}}\n\n"
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:"
)


def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def file_ext(path) -> str:
    return str(path).rsplit(".", 1)[-1].lower()


def read_table(path: str) -> pd.DataFrame:
    ext = file_ext(path)
    if ext == "jsonl":
        return pd.read_json(path, lines=True)
    if ext == "json":
        return pd.read_json(path)
    if ext == "tsv":
        return pd.read_csv(path, sep="\t")
    if ext == "csv":
        return pd.read_csv(path)
    if ext == "xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input extension: {ext}")


def dump_jsonl(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dream_video_index(value):
    value = str(value).replace("\\", "/").split("/")[-1]
    value = re.sub(r"\.mp4$", "", value, flags=re.IGNORECASE)
    return int(value) if value.isdigit() else None


def selected_rows(data: pd.DataFrame, args) -> pd.DataFrame:
    if args.indices:
        if "index" not in data.columns:
            raise ValueError("--indices requires an `index` column in the prediction file.")
        wanted = [int(x) for x in args.indices.split(",") if x.strip()]
        rows = data[data["index"].astype(int).isin(wanted)].copy()
    else:
        rows = data.iloc[args.start:args.start + args.limit].copy()
    if rows.empty:
        raise ValueError("No rows selected.")
    return rows


def normalize_rows(rows: pd.DataFrame, preserve_index: bool):
    rows = rows.copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in rows.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    if "index" not in rows.columns:
        rows["index"] = range(len(rows))

    if not preserve_index:
        mapped = rows["video"].map(dream_video_index)
        if mapped.isna().any():
            bad = rows.loc[mapped.isna(), "video"].tolist()
            raise ValueError(f"Cannot map DREAM video names to numeric ids: {bad[:3]}")
        rows["index"] = mapped.astype(int)

    if "question" not in rows.columns:
        rows["question"] = "Describe the video in detail."

    keep = ["index", "video", "question", "answer", "events", "prediction"]
    if "prompt" in rows.columns:
        keep.append("prompt")
    if "num_frames" in rows.columns:
        keep.append("num_frames")
    if "sampling_strategy" in rows.columns:
        keep.append("sampling_strategy")
    return rows[keep].to_dict("records")


def clean_judge_text(text: str) -> str:
    text = str(text).strip()
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```python"):
        text = text.replace("```python", "").replace("```", "").strip()
    text = text.replace("True", "true").replace("False", "false")
    if not text.startswith("{"):
        start = text.find("{")
        if start >= 0:
            text = text[start:]
    if not text.endswith("}"):
        end = text.rfind("}")
        if end >= 0:
            text = text[:end + 1]
    return text


def parse_judge_json(text: str, source: str) -> dict:
    text = clean_judge_text(text)
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    raise ValueError(f"Failed to parse {source}: {text[:300]}")


def parse_events_response(text: str, source: str):
    obj = parse_judge_json(text, source)
    events = obj.get("events")
    if events is None:
        for key, value in obj.items():
            if str(key).lower() == "events":
                events = value
                break
    if not isinstance(events, list):
        raise ValueError(f"Invalid {source}: missing list field `events`")
    return events


def parse_events_field(value):
    if is_empty(value):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        events = value.get("events")
        return events if isinstance(events, list) else None
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get("events"), list):
                return obj["events"]
        except Exception:
            pass
    return None


class AzureSDKJudge:
    def __init__(
        self,
        *,
        key=None,
        endpoint=None,
        deployment=None,
        api_version=None,
        max_completion_tokens=16384,
        retry=3,
        temperature=0,
    ):
        from openai import AzureOpenAI

        self.key = key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
        self.endpoint = (
            endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("ENDPOINT_URL")
            or os.getenv("AZURE_ENDPOINT")
        )
        self.deployment = (
            deployment
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or os.getenv("DEPLOYMENT_NAME")
            or os.getenv("AZURE_DEPLOYMENT_NAME")
        )
        self.api_version = (
            api_version
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or os.getenv("API_VERSION")
            or "2025-01-01-preview"
        )
        if not self.key:
            raise ValueError("Set AZURE_OPENAI_API_KEY or AZURE_API_KEY.")
        if not self.endpoint:
            raise ValueError("Set AZURE_OPENAI_ENDPOINT, ENDPOINT_URL, or AZURE_ENDPOINT.")
        if not self.deployment:
            raise ValueError(
                "Set AZURE_OPENAI_DEPLOYMENT_NAME, DEPLOYMENT_NAME, or AZURE_DEPLOYMENT_NAME."
            )

        self.client = AzureOpenAI(
            api_key=self.key,
            azure_endpoint=self.endpoint.rstrip("/"),
            api_version=self.api_version,
        )
        self.max_completion_tokens = max_completion_tokens
        self.retry = retry
        self.temperature = temperature

    def generate(self, message):
        if isinstance(message, str):
            prompt = message
        else:
            prompt = "\n".join(str(x.get("value", "")) for x in message if x.get("type") == "text")
        kwargs = dict(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        if self.max_completion_tokens and self.max_completion_tokens > 0:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        last_err = None
        for attempt in range(max(1, self.retry)):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as err:
                last_err = err
                if attempt + 1 < max(1, self.retry):
                    time.sleep(3)
        raise last_err


class MockJudge:
    def generate(self, message):
        prompt = message if isinstance(message, str) else message[0]["value"]
        if prompt.startswith("Bellow is a description"):
            return '{"events": ["subject performs an action."]}'
        return (
            '{"events": [{"event": "subject performs an action.", '
            '"relationship": "entailment", "reason": "mock"}]}'
        )


def build_judge(args):
    if args.backend == "mock":
        return MockJudge()
    if args.backend == "azure-sdk" or args.use_azure_sdk:
        return AzureSDKJudge(
            key=args.key,
            endpoint=args.azure_endpoint,
            deployment=args.azure_deployment_name,
            api_version=args.api_version,
            max_completion_tokens=args.max_completion_tokens,
            retry=args.retry,
        )
    try:
        from vlmeval.dataset.utils import build_judge as vlmeval_build_judge

        kwargs = dict(model=args.judge, retry=args.retry, verbose=args.verbose)
        if args.max_completion_tokens:
            kwargs["max_completion_tokens"] = args.max_completion_tokens
        kwargs.update(json.loads(args.judge_args))
        return vlmeval_build_judge(**kwargs)
    except Exception as err:
        if args.backend == "vlmeval":
            raise
        print(f"[DREAM smoke] VLMEval build_judge unavailable, falling back to Azure SDK: {err}")
        return AzureSDKJudge(
            key=args.key,
            endpoint=args.azure_endpoint,
            deployment=args.azure_deployment_name,
            api_version=args.api_version,
            max_completion_tokens=args.max_completion_tokens,
            retry=args.retry,
        )


def call_judge(judge, stage: str, prompt: str, log: list):
    response = judge.generate([{"type": "text", "value": prompt}])
    log.append({"stage": stage, "prompt": prompt, "response": response})
    return response


def extract_events(judge, caption: str, log: list, stage: str):
    response = call_judge(judge, stage, EXTRACTION_PROMPT.format(caption=caption), log)
    return parse_events_response(response, stage)


def evaluate_relationship(judge, events, prediction: str, reference: str, log: list, stage: str):
    events_for_prompt = events if events else [reference.replace("\n", " ")]
    prompt = RELATIONSHIP_PROMPT.format(prediction=prediction, events=str(events_for_prompt))
    response = call_judge(judge, stage, prompt, log)
    return parse_events_response(response, stage)


def process_row(row: dict, judge):
    idx = int(row["index"])
    gt_response = str(row["answer"]).lower()
    prediction = str(row["prediction"]).lower()
    judge_io = []
    try:
        gt_events = parse_events_field(row.get("events"))
        if gt_events is None:
            gt_events = extract_events(judge, gt_response, judge_io, "gt_event_extraction")
        pred_events = extract_events(judge, prediction, judge_io, "prediction_event_extraction")

        recall_events = evaluate_relationship(
            judge, gt_events, prediction, gt_response, judge_io, "recall_relationship"
        )
        recall_hits = sum(
            1 for e in recall_events
            if isinstance(e, dict) and str(e.get("relationship", "")).lower() == "entailment"
        )
        score_r = recall_hits / len(gt_events) if gt_events else 1.0

        precision_events = evaluate_relationship(
            judge, pred_events, gt_response, prediction, judge_io, "precision_relationship"
        )
        precision_hits = sum(
            1 for e in precision_events
            if isinstance(e, dict) and str(e.get("relationship", "")).lower() == "entailment"
        )
        score_p = precision_hits / len(pred_events) if pred_events else 1.0

        f1 = 2 * score_r * score_p / (score_r + score_p) if score_r + score_p > 0 else 0.0
        return {
            **row,
            "score_r": score_r,
            "score_p": score_p,
            "f1": f1,
            "gt_events": gt_events,
            "pred_events": pred_events,
            "recall_judge_events": recall_events,
            "precision_judge_events": precision_events,
            "judge_io": judge_io,
        }
    except Exception as err:
        return {
            **row,
            "score_r": -1,
            "score_p": -1,
            "f1": -1,
            "error": str(err),
            "judge_io": judge_io,
        }


def summarize(results):
    valid = [r for r in results if r.get("score_r", -1) >= 0]
    if not valid:
        return {"Overall": {"Recall": "0.0000", "Precision": "0.0000", "F1": "0.0000"},
                "num_valid": 0, "num_total": len(results)}
    avg_r = sum(r["score_r"] for r in valid) / len(valid)
    avg_p = sum(r["score_p"] for r in valid) / len(valid)
    f1 = 2 * avg_r * avg_p / (avg_r + avg_p) if avg_r + avg_p > 0 else 0.0
    return {
        "Overall": {"Recall": f"{avg_r:.4f}", "Precision": f"{avg_p:.4f}", "F1": f"{f1:.4f}"},
        "num_valid": len(valid),
        "num_total": len(results),
    }


def run_rows(rows, judge, nproc: int):
    if nproc <= 1:
        return [process_row(row, judge) for row in rows]
    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        future_to_idx = {executor.submit(process_row, row, judge): i for i, row in enumerate(rows)}
        for future in as_completed(future_to_idx):
            results[future_to_idx[future]] = future.result()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-file", required=True, help="Existing DREAM prediction xlsx/tsv/json/jsonl.")
    parser.add_argument("--output-dir", default="/tmp/dream1k_judge_smoke")
    parser.add_argument("--prefix", default="dream1k_5pred")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indices", default="", help="Comma-separated existing indices instead of start/limit.")
    parser.add_argument("--preserve-index", action="store_true",
                        help="Keep file index. Default maps index from video/NNN.mp4 for DREAM parity.")
    parser.add_argument("--judge", default="gpt-4o")
    parser.add_argument("--backend", choices=("auto", "azure-sdk", "vlmeval", "mock"), default="auto")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-azure-sdk", action="store_true",
                        help="Compatibility flag; equivalent to --backend azure-sdk.")
    parser.add_argument("--key", default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=16384,
                        help="Set 0 to omit max_completion_tokens.")
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-deployment-name", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--judge-args", default="{}", help="Extra JSON kwargs for VLMEval build_judge backend.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = normalize_rows(selected_rows(read_table(args.pred_file), args), args.preserve_index)
    input_jsonl = output_dir / f"{args.prefix}_input.jsonl"
    score_jsonl = output_dir / f"{args.prefix}_{args.judge}_score.jsonl"
    rating_json = output_dir / f"{args.prefix}_{args.judge}_rating.json"
    judge_io_jsonl = output_dir / f"{args.prefix}_{args.judge}_judge_io.jsonl"

    if args.force:
        for path in (input_jsonl, score_jsonl, rating_json, judge_io_jsonl):
            if path.exists():
                path.unlink()

    dump_jsonl(rows, input_jsonl)
    judge = build_judge(args)

    print(f"[DREAM smoke] pred source: {args.pred_file}")
    print(f"[DREAM smoke] selected input: {input_jsonl}")
    print(f"[DREAM smoke] rows: {len(rows)}")
    print(f"[DREAM smoke] backend: {'azure-sdk' if args.use_azure_sdk else args.backend}")

    results = run_rows(rows, judge, args.nproc)
    rating = summarize(results)

    dump_jsonl(results, score_jsonl)
    dump_jsonl(
        [
            {
                "index": r.get("index"),
                "score_r": r.get("score_r"),
                "score_p": r.get("score_p"),
                "f1": r.get("f1"),
                "error": r.get("error"),
                "judge_io": r.get("judge_io", []),
            }
            for r in results
        ],
        judge_io_jsonl,
    )
    rating_json.write_text(json.dumps(rating, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DREAM smoke] result:")
    print(json.dumps(rating, ensure_ascii=False, indent=2))
    print(f"[DREAM smoke] score jsonl: {score_jsonl}")
    print(f"[DREAM smoke] rating json: {rating_json}")
    print(f"[DREAM smoke] judge IO jsonl: {judge_io_jsonl}")


if __name__ == "__main__":
    main()

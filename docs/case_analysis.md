# Case Analysis System

This note is for the situation where benchmark scores are already available, but the main question is:

- which ability improved;
- where the improvement is concentrated;
- how to inspect only a few cases instead of manually reading many examples.

The recommended workflow is a three-layer pipeline:

1. Macro layer: compare model-level benchmark deltas.
2. Subgroup layer: split each benchmark by native metadata such as `category`, `source`, `area`, `reasoning`, or `tag`.
3. Case layer: only keep a few representative swing cases for the strongest subgroup deltas.

That keeps the reading order small:

- first look at charts;
- then inspect only the strongest subgroup changes;
- only then read a handful of cases.

## Why this is better than browsing raw cases

Reading raw examples directly is expensive and noisy. A good case system should answer:

- "Did temporal grounding improve, or only multiple-choice parsing?"
- "Is the gain concentrated on one subgroup?"
- "Did the new model introduce a regression somewhere else?"

So the case system should not start from examples. It should start from deltas, then use cases as evidence.

## Minimal data contract

The lightweight script in [scripts/build_case_report.py](/Users/lostgreen/Desktop/Codes/VideoProxy/eval/scripts/build_case_report.py) works on row-level `_score` files produced by existing evaluators.

It currently works best when the saved score file includes one of:

- `score`
- `correct`
- `iou`

This already covers many video benchmarks in the current repo, including exact-match style MCQ benchmarks and temporal-grounding style benchmarks.

## Output structure

The script generates:

- `01_dataset_delta.png`: dataset-level score delta chart
- `02_case_balance.png`: candidate-only wins vs baseline-only wins at dataset level
- `03_group_delta.png`: strongest subgroup deltas
- `report.md`: compact human-readable report
- `dataset_summary.csv`: dataset-level summary table
- `group_summary.csv`: subgroup-level summary table
- `case_inventory.jsonl`: normalized full case inventory
- `case_inventory.csv`: spreadsheet-friendly export of the same inventory
- `representative_cases.jsonl`: only the selected high-signal cases shown in the report
- `ai_prompt.md`: a ready-to-use prompt for handing selected cases to another AI for narrative analysis
- `summary.json`: structured output for later dashboard integration

## Usage

```bash
python scripts/build_case_report.py \
  --work-dir /path/to/eval_outputs \
  --baseline MODEL_A \
  --candidate MODEL_B \
  --out-dir ./case_report
```

Optional filters:

```bash
python scripts/build_case_report.py \
  --work-dir /path/to/eval_outputs \
  --baseline MODEL_A \
  --candidate MODEL_B \
  --data AoTBench_ReverseFilm_16frame FutureOmni_64frame \
  --min-group-size 6 \
  --top-groups 8 \
  --cases-per-group 2 \
  --out-dir ./case_report
```

## Recommended reading pattern

Use the report in this order:

1. Read `01_dataset_delta.png` to see where score movement is real.
2. Read `02_case_balance.png` to see whether the gain comes from many repaired cases or only a few swings.
3. Read `03_group_delta.png` to identify the most concentrated ability shifts.
4. Open `report.md` and only inspect the subgroup sections with the biggest delta.
5. If you want natural-language interpretation, feed `ai_prompt.md` to an LLM.

## Practical rule for keeping the system light

Do not show too many cases.

A good default is:

- per subgroup: at most `2` gain cases and `1` regression case, or vice versa
- per report: only top `6-10` subgroup changes
- always keep one regression block, even if the candidate model is better overall

This is enough to make capability differences legible without turning the report into a case dump.

## Suggested next step

If this report becomes part of the regular workflow, the next reasonable integration is:

1. keep `case_inventory.jsonl` and `summary.json` as the standard interchange files;
2. add a new "Case Analysis" tab in the existing `vis/` dashboard;
3. read subgroup charts first, and make representative cases collapsible under each subgroup.

That gives you a system where the default experience is chart-first, not case-first.

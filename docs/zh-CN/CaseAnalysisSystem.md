# Case 整理系统

这套系统面向一个很具体的问题：

- benchmark 分数已经有了；
- 你不只是想看总分；
- 你想知道模型到底提升了什么能力，退化了什么能力；
- 你不想靠人工翻几百个 case 来猜。

所以它的设计原则不是“先看 case”，而是：

1. 先看数据集级别的分数变化；
2. 再看能力子组在哪里发生集中变化；
3. 最后只读少量代表性 case，把变化当作证据。

## 核心思路

系统把两个模型在同一 benchmark 上的行级结果合并成一张标准化 case 表，然后给每个 case 打上状态标签：

- `candidate_fix`: 新模型修复了旧模型错误
- `candidate_drop`: 新模型在这个 case 上退化
- `stable_correct`: 两个模型都做对
- `stable_wrong`: 两个模型都做错
- `candidate_gain`: 连续值指标上新模型更好
- `stable_tie`: 连续值指标上两者持平

这样，case 就不再是零散样本，而是可以被聚合、统计、筛选的分析对象。

## 输出产物

运行 [scripts/build_case_report.py](/Users/lostgreen/Desktop/Codes/VideoProxy/eval/scripts/build_case_report.py) 后，会得到一组固定产物：

- `01_dataset_delta.png`
  用来看哪些 benchmark 的总分真的动了。
- `02_case_balance.png`
  用来看每个 benchmark 里到底是“新模型修复了更多 case”，还是“同时也引入了不少回退”。
- `03_group_delta.png`
  用来看能力变化最集中的 subgroup。
- `report.md`
  适合人直接读，按“总分 -> subgroup -> 代表 case”的顺序组织。
- `dataset_summary.csv`
  数据集层面的汇总表。
- `group_summary.csv`
  subgroup 层面的汇总表，适合继续排序、过滤、画图。
- `case_inventory.jsonl`
  全量 case 标准表，适合二次分析、喂给 LLM、接 dashboard。
- `case_inventory.csv`
  方便直接用表格软件筛选。
- `representative_cases.jsonl`
  只保留高信号 subgroup 下的代表 case。
- `summary.json`
  机器可读入口，方便后续接可视化页面。

## 推荐阅读顺序

1. 先看 `01_dataset_delta.png`
   判断变化发生在哪些 benchmark，而不是先陷入样本细节。
2. 再看 `02_case_balance.png`
   判断提升是不是靠大量修复 case 带来的，还是只靠少数 swing case。
3. 再看 `03_group_delta.png`
   找到最可能对应“能力变化”的 subgroup。
4. 最后读 `report.md`
   只打开变化最大的 subgroup，看里面的代表 case。
5. 如果想做更细分析，再筛 `case_inventory.jsonl/csv`
   例如专门看 `candidate_fix`、`stable_wrong`、某个 `reasoning` 类别。

## 这套系统回答什么问题

它最适合回答下面这些问题：

- “模型提升主要发生在时序理解，还是只是选择题抽取更稳了？”
- “提升是全面的，还是只集中在某个 benchmark 的某个子类上？”
- “新模型虽然总分更高，但有没有明显的回退区？”
- “哪些 case 值得拿出来做人类复盘或喂给分析模型？”

## 使用方式

最基本的用法：

```bash
python scripts/build_case_report.py \
  --work-dir /path/to/eval_outputs \
  --baseline MODEL_A \
  --candidate MODEL_B \
  --out-dir ./case_report
```

只分析部分 benchmark：

```bash
python scripts/build_case_report.py \
  --work-dir /path/to/eval_outputs \
  --baseline MODEL_A \
  --candidate MODEL_B \
  --data AoTBench_ReverseFilm_16frame FutureOmni_64frame \
  --out-dir ./case_report
```

强制只按指定字段做 subgroup：

```bash
python scripts/build_case_report.py \
  --work-dir /path/to/eval_outputs \
  --baseline MODEL_A \
  --candidate MODEL_B \
  --group-columns category reasoning subtask \
  --min-group-size 6 \
  --cases-per-group 2 \
  --out-dir ./case_report
```

## 推荐落地方式

如果你后面想把这件事变成长期流程，建议按下面的方式沉淀：

1. 把 `case_inventory.jsonl` 当成标准 case 中间层；
2. 每次新模型出结果后，先产出这份标准表；
3. 后续所有 dashboard、复盘、人工抽样都基于这份表做；
4. 真正人工阅读时，只看 `representative_cases.jsonl` 和 `report.md`；
5. 如果某些 benchmark 有稳定的能力字段，再补 dataset-specific 的 subgroup 配置。

这样你最终得到的不是“很多截图和零散 case”，而是一套可持续比较模型能力变化的整理系统。

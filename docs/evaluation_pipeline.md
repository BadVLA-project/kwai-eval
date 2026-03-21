# 评测流程文档：CoT vs 非 CoT

## 1. 脚本总览

| 脚本 | 用途 | USE_COT | 加速参数 |
|------|------|---------|----------|
| `run_no_cot.sh` | 不带 CoT 的直接回答模式 | `0` | 无额外设置 |
| `run_cot.sh` | 带 CoT 的推理链模式 | `1` | `VLLM_MAX_NUM_SEQS=32` / `VLLM_BATCH_CHUNK_SIZE=64` / `VLLM_GPU_MEMORY_UTILIZATION=0.90` |
| `run.sh` | 通用脚本（通过 `USE_COT` 环境变量切换） | 可配置 | 无预设 |

三个脚本评测的 **模型** 和 **数据集** 完全一致：

**模型（4个消融实验）：**
- `Qwen3-VL-4B-Instruct_aot_ablation_exp1_v2t_binary`
- `Qwen3-VL-4B-Instruct_aot_ablation_exp2_v2t_3way`
- `Qwen3-VL-4B-Instruct_aot_ablation_exp3_t2v_binary`
- `Qwen3-VL-4B-Instruct_aot_ablation_exp4_t2v_3way`

**数据集（9个）：**
- AoTBench_ReverseFilm_16frame, AoTBench_UCF101_16frame, AoTBench_Rtime_t2v_16frame
- AoTBench_Rtime_v2t_16frame, AoTBench_QA_16frame
- FutureOmni_64frame, CharadesTimeLens_1fps, MVBench_MP4_1fps, PerceptionTest_val_16frame

---

## 2. 完整调用链

```
run_cot.sh / run_no_cot.sh
│
│  export USE_COT=1/0
│  torchrun --nproc-per-node=N run.py --use-vllm --data ... --model ...
│
├─► run.py (每个 rank 一个进程)
│   ├── 启动时按 LOCAL_RANK 切分 CUDA_VISIBLE_DEVICES（每 rank 1 GPU）
│   ├── dist.init_process_group(backend='nccl')
│   │
│   ├── for model_name in args.model:       ← 模型串行
│   │   └── for dataset_name in args.data:  ← 数据集串行
│   │       └── infer_data_job_video(model, ..., use_vllm=True)
│   │
│   └── 评测阶段（judge）
│
├─► infer_data_job_video()  [inference_video.py]
│   └── infer_data(model, ..., use_vllm=True)
│       │
│       ├── 模型构建: supported_VLM[model_name](use_vllm=True)
│       │   └── Qwen3VLChat.__init__(use_vllm=True)
│       │       ├── 读取 USE_COT 环境变量
│       │       ├── 创建 vLLM LLM 实例 (tp=1 for 4B)
│       │       └── 设置采样参数
│       │
│       ├── 【vLLM 批量路径】(use_vllm=True)
│       │   ├── 构建全部 prompts
│       │   └── model.generate_batch_vllm(batch_structs, chunk_size=N)
│       │       └── vLLM continuous batching 推理
│       │
│       └── 合并各 rank 结果 → 保存 result_file
```

---

## 3. CoT 与非 CoT 的参数差异

### 3.1 模型初始化阶段 (`Qwen3VLChat.__init__`)

| 参数 | 非 CoT (USE_COT=0) | CoT (USE_COT=1) |
|------|---------------------|------------------|
| `temperature` | **0.7**（config.py 原值） | **0.7**（CoT 强制覆盖） |
| `do_sample` | `True`（因 temp > 0） | `True`（CoT 强制） |
| `max_new_tokens` | **16384**（config.py 原值） | **2048**（CoT 覆盖） |
| `post_prompt` | `None` | `'Please think step by step inside <think> tags, then provide the final answer inside <answer> tags.'` |
| `extract_think_answer` | `False` | `True` |

> **注意**：4个消融模型在 config.py 中已设置 `temperature=0.7`，所以 CoT 和非 CoT
> 模式下 temperature 实际相同。区别主要在 **prompt 改写** 和 **输出提取**。

### 3.2 Prompt 改写 (`_rewrite_prompt_for_cot`)

当 `USE_COT=1` 时，`_prepare_content()` 会在构建 prompt 后调用 `_rewrite_prompt_for_cot()`：

1. **删除**"直接回答"类的 benchmark 指令，例如：
   - `"Answer with the option's letter from the given choices directly."`
   - `"Only give the best option."`
   - `"Respond with only the letter (A/B/C...) of the correct option."`
2. **删除** assistant 角色的预填充消息（如 MVBench 的 `"Best option:("`）
3. **追加** CoT 指令：`"Please think step by step inside <think> tags, then provide the final answer inside <answer> tags."`

### 3.3 输出后处理

- **非 CoT**：直接使用模型生成文本
- **CoT**：从生成文本中提取 `<answer>...</answer>` 内容作为最终答案
  - 如果找不到 `<answer>` 标签，则 fallback：删除 `<think>...</think>` 后用剩余文本

---

## 4. vLLM 推理路径详解

### 4.1 为什么需要 `--use-vllm`

`config.py` 中已有 `use_vllm=True`，但 `inference_video.py` 中模型构建时会传入 `kwargs = {'use_vllm': use_vllm}`，其中 `use_vllm` 来自 CLI 参数。调用时 **kwargs 覆盖 partial 参数**，因此 `--use-vllm` 是必需的。

### 4.2 GPU 分配与数据并行

```
torchrun --nproc-per-node=N
│
├── Rank 0: CUDA_VISIBLE_DEVICES=0  → vLLM(tp=1, GPU 0) → 处理 1/N 数据
├── Rank 1: CUDA_VISIBLE_DEVICES=1  → vLLM(tp=1, GPU 1) → 处理 1/N 数据
├── Rank 2: CUDA_VISIBLE_DEVICES=2  → vLLM(tp=1, GPU 2) → 处理 1/N 数据
└── ...
```

- 4B 模型自动设置 `tp=1`（`_default_tp_size`: <8B → tp=1）
- 每个 rank 独立加载模型，独立做 vLLM 推理，零通信开销
- 模型约 10GB 显存，80GB 卡有充足余量给 KV cache

### 4.3 批量推理流程

进入 vLLM batch path 后（`inference_video.py`）：

```
1. 收集当前 rank 的全部剩余样本 prompts → batch_structs
2. model.generate_batch_vllm(batch_structs, chunk_size=N)
   ├── 分 chunk 处理（避免一次性加载太多视频帧到内存）
   ├── 每个 chunk: 构建 vLLM requests → llm.generate(reqs)
   │   └── vLLM 内部 continuous batching (max_num_seqs 控制并发)
   └── 收集结果
3. 保存到 pkl 文件
```

### 4.4 CoT 加速参数（80GB 显存优化）

| 环境变量 | 默认值 | CoT 加速值 | 说明 |
|----------|--------|------------|------|
| `VLLM_MAX_NUM_SEQS` | 8 | **32** | vLLM 同时处理的序列数。4B 模型显存占用低，可大幅增加 |
| `VLLM_BATCH_CHUNK_SIZE` | 32 | **64** | 每次构建的 prompt 数量。增大减少构建/推理切换开销 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.85 | **0.90** | vLLM 可使用的显存比例。更多显存 = 更大 KV cache pool |

**加速原理**：
- 4B 模型权重 ≈ 10GB，80GB 显存剩余 ~70GB 可用于 KV cache
- CoT `max_new_tokens=2048`，每个序列的 KV cache 较小
- 增大 `max_num_seqs` 到 32，vLLM 可以同时处理 32 个序列
- vLLM continuous batching 自动调度：一个序列 decode 完毕即填入新序列
- 有效消除 GPU idle time，显著提高吞吐

---

## 5. 评测阶段（Judge）

推理完成后，`run.py` 会根据数据集类型自动选择 judge：

| 数据集 | Judge 类型 |
|--------|-----------|
| MVBench, LongVideoBench | `exact_matching`（精确匹配） |
| MCQ / Y/N / Video-MCQ 类 | `chatgpt-0125`（API judge） |
| AoTBench 系列 | 取决于数据集 TYPE |

CoT 和非 CoT 模式使用 **相同的评测 judge**。评测时比较的是最终答案文本（CoT 模式经过 `<answer>` 提取后）。

---

## 6. 注意事项 & 潜在问题

### ⚠️ 消融模型的 temperature 已为 0.7

4 个消融模型在 `config.py` 中设置了 `temperature=0.7`、`do_sample=True`。
这意味着 **非 CoT 模式也是采样模式（非 greedy）**。CoT 覆盖的 temperature 恰好也是 0.7，实际上两种模式的采样参数一致。

两种模式的核心区别仅在于：
1. **Prompt**：CoT 追加了 step-by-step 指令，删除了"直接回答"指令
2. **max_new_tokens**：16384 → 2048（CoT 模式反而更短）
3. **输出后处理**：CoT 提取 `<answer>` 标签内容

### ⚠️ 多 GPU 时 MASTER_PORT 冲突

如果同一节点上同时运行 `run_cot.sh` 和 `run_no_cot.sh`，需要设置不同的 `MASTER_PORT`：
```bash
# 终端 1
MASTER_PORT=29500 bash run_no_cot.sh
# 终端 2
MASTER_PORT=29501 bash run_cot.sh
```

### ⚠️ 输出目录隔离

两种模式的输出目录都是 `$WORK_DIR/$MODEL_NAME/T{date}_G{commit}`。
由于 eval_id 包含日期和 git hash，同一天如果连续运行两种模式，结果会写到同一目录。
如需隔离，建议设置不同的 `WORK_DIR`：
```bash
WORK_DIR=.../evaluation_cot bash run_cot.sh
WORK_DIR=.../evaluation_no_cot bash run_no_cot.sh
```

---

## 7. 快速使用

```bash
# 不带 CoT 评测
bash run_no_cot.sh

# 带 CoT 评测（含 80GB 加速）
bash run_cot.sh

# 自定义参数
NGPU=4 REUSE=1 MASTER_PORT=29501 bash run_cot.sh

# 覆盖加速参数
VLLM_MAX_NUM_SEQS=64 VLLM_BATCH_CHUNK_SIZE=128 bash run_cot.sh
```

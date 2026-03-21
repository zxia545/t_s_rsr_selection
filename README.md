# Teacher Selection Pipeline

This directory contains a standalone pipeline for selecting the best teacher-output dataset with either Rank Surprisal Ratio (RSR) or gradient baselines such as GRACE.

It is designed for the workflow you described:

1. Each teacher has one `.jsonl` or `.json` file.
2. Each record contains fields such as `system`, `instruction`, and `teacher_output`.
3. The pipeline samples a shared subset of examples across all teachers.
4. A student model scores every teacher's sampled responses with the configured selection metric.
5. Teachers are ranked according to that metric.
6. The best teacher file is copied out as a final dataset and a matching `dataset_info.json` is produced.

## Files

- `teacher_rsr_selection.py`
  - Single-run entrypoint.
  - One teacher folder + one student model.
  - Handles shared-id sampling, multi-worker GPU scheduling, scoring, ranking, and final dataset export.

- `rsr_core.py`
  - Core scoring implementation.
  - Loads the student model/tokenizer, builds chat inputs, computes RSR, and extracts projected student gradients for GRACE / G-Norm / G-Vendi.

- `rsr_pipeline.py`
  - Batch runner for multiple datasets and multiple student models.
  - Uses a JSON config and dispatches repeated runs of `teacher_rsr_selection.py`.

- `GRACE_BASELINE.md`
  - Detailed note for the GRACE / G-Norm / G-Vendi teacher-selection path.
  - Covers inputs, outputs, formulas, paper alignment, and output artifacts.

- `run_rsr_pipeline.sh`
  - Thin shell wrapper around `rsr_pipeline.py`.

- `configs/rsr_pipeline.example_grace.json`
  - Explicit GRACE batch config.

- `configs/rsr_pipeline.example_rsr.json`
  - Explicit RSR batch config.

- `configs/rsr_pipeline_example_test_grace.json`
  - Tiny GRACE dry-run config.

- `configs/rsr_pipeline_example_test_rsr.json`
  - Tiny RSR dry-run config.

- `configs/rsr_pipeline.example.json`
  - Backward-compatible alias of the GRACE example config.

- `requirements.txt`
  - Minimal Python dependencies for this root-level pipeline.

## Data Assumption

Each teacher file should contain the same task set identified by a shared id field.

Example record:

```json
{
  "id": 48,
  "instruction": "Mike has earned a total of $160 in wages this week...",
  "system": "You are a helpful assistant.\nPlease answer the math question below exactly as asked.",
  "teacher_output": "$88"
}
```

For this kind of dataset, the pipeline converts each sampled record into:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

The default field mapping is:

- `id_field = id`
- `system_field = system`
- `prompt_field = instruction`
- `response_field = teacher_output`

These can be overridden per dataset or per run.

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Current root-level requirements pin:

- `transformers==4.44.2`

That pin is intentional because this machine was validated with `torch>=2.0.0`.

## Single Run

Run one dataset folder against one student model with GRACE:

```bash
python teacher_rsr_selection.py \
  --teacher-folder dataset/gsm8k_teacher_output \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --model-name qwen2.5-0.5b-instruct \
  --selection-metric grace \
  --teacher-glob "*.jsonl" \
  --id-field id \
  --system-field system \
  --prompt-field instruction \
  --response-field teacher_output \
  --sample-size 200 \
  --seed 42 \
  --batch-size 1 \
  --dtype float16 \
  --chat-template qwen \
  --max-model-len 2048 \
  --gpus-per-worker 1 \
  --output-root outputs/gsm8k_qwen05b \
  --copy-selected-to-output
```

For the original RSR selector, use `--selection-metric rsr`.

### Supported Metrics

- `rsr`
  - Rank Surprisal Ratio on sampled responses.

- `grace`
  - Dataset-level GRACE score from the RSR paper setup: sample 200 trajectories, compute one projected student gradient per sample, split into folds, and average `Tr(Σ_hat_bg^{-1} Σ_test)`.

- `g_norm`
  - Gradient-norm baseline computed from the same processed gradients.

- `g_vendi`
  - Gradient diversity baseline computed from the normalized gradient spectrum.

### GRACE Notes

- Use `--batch-size 1`.
- Recommended paper-style settings are `--sample-size 200`, `--grace-projection-dim 512`, `--grace-num-partitions 10`.
- The implementation keeps the paper structure but uses a fixed sparse CountSketch-style random projection instead of a dense `d x D` Rademacher matrix so large-model gradients remain tractable.
- GRACE is dataset-level only. The pipeline writes per-sample diagnostics, but teacher ranking uses the dataset score.

### Important Arguments

- `--teacher-folder`
  - Folder containing one teacher file per teacher.

- `--teacher-glob`
  - File pattern used to discover teacher files.

- `--id-field`
  - Shared unique example id across teacher files.

- `--system-field`
  - Field used for the system prompt. Empty values are allowed.

- `--prompt-field`
  - Field used as the user prompt.

- `--response-field`
  - Field used as the assistant response to be scored.

- `--selection-metric`
  - One of `rsr`, `grace`, `g_norm`, or `g_vendi`.

- `--sample-size`
  - Number of shared ids to sample across teachers.

- `--grace-projection-dim`
  - Output dimension for projected gradients when using gradient baselines.

- `--grace-num-partitions`
  - Number of cross-validation partitions for GRACE.

- `--grace-smoothing`
  - Smoothing coefficient `nu` used in the regularized inverse covariance.

- `--gpus-per-worker`
  - Number of visible GPUs assigned to each scoring worker.

- `--max-workers`
  - Optional hard cap on worker count. `0` means auto.

- `--dataset-info-path`
  - Global dataset registry file to update.

- `--dataset-export-dir`
  - Location where the winning teacher file is copied as a dataset artifact.

- `--output-root`
  - Run output directory. This is the main folder you inspect after the run.

## Output Structure

A single run creates a folder like:

```text
outputs/gsm8k_qwen05b/
  final_dataset/
    <metric>_selected__...jsonl
    dataset_info.json
  logs/
  manifests/
  prepared_messages/
  sample_metrics/
  worker_outputs/
  run_summary.json
  sampled_ids.json
  teacher_overview.json
  teacher_ranking.json
  teacher_ranking.tsv
```

### Key Outputs

- `teacher_ranking.json`
  - Full ranking sorted by the configured selection metric.

- `teacher_ranking.tsv`
  - Same ranking in tabular form.

- `run_summary.json`
  - Run metadata, selected teacher, GPU assignment, and output paths.

- `final_dataset/<selected>.jsonl`
  - Copy of the winning original teacher file.

- `final_dataset/dataset_info.json`
  - A minimal dataset registry containing only the selected dataset entry.

- `dataset_export_dir/<selected>.jsonl`
  - Copy of the winning file under your shared dataset export location.

- `dataset_info_path`
  - The global dataset registry is updated with the selected dataset entry.

## Batch Pipeline

Use `rsr_pipeline.py` when you want to score many datasets and many student models.

```bash
python rsr_pipeline.py \
  --config configs/rsr_pipeline.example_grace.json \
  --output-root outputs/pipeline_runs
```

or

```bash
bash run_rsr_pipeline.sh configs/rsr_pipeline.example_grace.json --output-root outputs/pipeline_runs
```

For the RSR selector, switch to:

```bash
python rsr_pipeline.py \
  --config configs/rsr_pipeline.example_rsr.json \
  --output-root outputs/pipeline_runs
```

## Config Format

The JSON config has four top-level sections:

- `defaults`
  - Shared defaults for all datasets/models/runs.

- `models`
  - Student model definitions.

- `datasets`
  - Teacher dataset definitions.

- `runs`
  - Optional explicit run matrix.

If `runs` is omitted, the pipeline uses the full dataset × model cross product.

### Merge Priority

Config values are merged in this order:

`defaults` -> `datasets[i]` -> `runs[i]`

That means dataset-specific field mappings are supported directly in each dataset block.

### Recommended Setup

Keep separate config files for `grace` and `rsr`.

That keeps:

- `selection_metric`
- metric-specific parameters
- export directories
- dataset entry prefixes

from getting mixed together.

In this repo:

- `configs/rsr_pipeline.example_grace.json`
- `configs/rsr_pipeline.example_rsr.json`
- `configs/rsr_pipeline_example_test_grace.json`
- `configs/rsr_pipeline_example_test_rsr.json`

The older files `configs/rsr_pipeline.example.json` and `configs/rsr_pipeline_example_test.json` are kept as GRACE aliases for backward compatibility.

### Example Config

```json
{
  "defaults": {
    "id_field": "id",
    "system_field": "system",
    "prompt_field": "instruction",
    "response_field": "teacher_output",
    "selection_metric": "grace",
    "sample_size": 200,
    "seed": 42,
    "batch_size": 1,
    "dtype": "float16",
    "chat_template": "qwen",
    "rank_clip_r": 100,
    "grace_projection_dim": 512,
    "grace_num_partitions": 10,
    "grace_smoothing": 0.0001,
    "max_model_len": 2048,
    "gpus_per_worker": 1,
    "max_workers": 0,
    "copy_selected_to_output": true
  },
  "models": [
    {
      "name": "qwen2.5-0.5b-instruct",
      "model_path": "Qwen/Qwen2.5-0.5B-Instruct"
    }
  ],
  "datasets": [
    {
      "name": "gsm8k_teacher_output",
      "teacher_folder": "dataset/gsm8k_teacher_output",
      "teacher_glob": "*.jsonl",
      "dataset_info_path": "dataset/dataset_info.json",
      "dataset_export_dir": "dataset/grace_selected_teachers",
      "dataset_entry_prefix": "grace_selected"
    },
    {
      "name": "other_dataset",
      "teacher_folder": "dataset/other_teacher_output",
      "id_field": "_id",
      "system_field": "sys_prompt",
      "prompt_field": "question",
      "response_field": "answer"
    }
  ],
  "runs": [
    {
      "name": "gsm8k_qwen05b",
      "dataset": "gsm8k_teacher_output",
      "model": "qwen2.5-0.5b-instruct"
    }
  ]
}
```

## Multi-GPU Behavior

`teacher_rsr_selection.py` auto-detects visible GPUs from:

- `CUDA_VISIBLE_DEVICES`, if set
- otherwise `torch.cuda.device_count()`

Workers are formed by grouping visible GPUs according to `--gpus-per-worker`.

Examples:

- 4 visible GPUs, `--gpus-per-worker 1`
  - 4 workers
  - each worker scores a subset of teachers on 1 GPU

- 4 visible GPUs, `--gpus-per-worker 2`
  - 2 workers
  - each worker loads the model across 2 GPUs

- `--max-workers 2`
  - caps the number of launched workers even if more GPUs are visible

Teacher files are assigned to workers in round-robin order.

## Selection Logic

For each run:

1. Discover all teacher files.
2. Validate records using the configured `id/prompt/response` fields.
3. Intersect valid ids across all teachers.
4. Sample `sample_size` shared ids with the configured seed.
5. Convert records into chat-format messages.
6. Score each teacher with the student model using the configured selection metric.
7. Rank teachers according to that metric's objective.
8. Copy the best teacher file into output locations.
9. Write ranking artifacts and dataset info files.

- For `rsr`, smaller `rank_surprisal_ratio` is better.
- For `grace`, smaller `grace` is better.

## Dry Run and Filtering

`rsr_pipeline.py` supports:

```bash
python rsr_pipeline.py \
  --config configs/rsr_pipeline.example_grace.json \
  --output-root outputs/pipeline_runs \
  --only-models qwen2.5-0.5b-instruct \
  --only-datasets gsm8k_teacher_output \
  --dry-run
```

Useful flags:

- `--only-models`
  - Restrict to a subset of model names.

- `--only-datasets`
  - Restrict to a subset of dataset names.

- `--dry-run`
  - Print commands without executing them.

- `--fail-fast`
  - Stop the batch immediately on the first failed run.

## Notes and Limitations

- The root-level implementation is focused on text chat inputs.
- It expects data that can be converted into `system/user/assistant` messages.
- It does not include the old repo's multimodal processing path.
- It does not currently save model inference caches in parquet/jsonl the way the legacy `RankSurprisalRatio` tools can.
- `chat_template` currently supports marker detection for `qwen` and `llama3`.

## Recommended Workflow

For recurring experiments:

1. Put all student models in `models`.
2. Put each teacher-output dataset in `datasets`.
3. Put dataset-specific field mappings directly inside each dataset block.
4. Use `runs` only when you need custom output names or per-run overrides.
5. Launch everything through `rsr_pipeline.py`.

That keeps the code stable and moves experiment control into config.

# Teacher Selection Pipeline

## Files

- `teacher_rsr_selection.py`
  - single run entrypoint
  - input: one teacher-output folder + one student model
  - output: ranking files + `final_dataset/` with selected jsonl and `dataset_info.json`

- `rsr_pipeline.py`
  - batch runner for multiple datasets and multiple student models
  - reads a JSON config

- `GRACE_BASELINE.md`
  - detailed explanation of the GRACE baseline used by the selector

- `run_rsr_pipeline.sh`
  - thin shell wrapper around `rsr_pipeline.py`

- `configs/rsr_pipeline.example_grace.json`
  - explicit GRACE batch config

- `configs/rsr_pipeline.example_rsr.json`
  - explicit RSR batch config

- `configs/rsr_pipeline_example_test_grace.json`
  - tiny GRACE dry-run config

- `configs/rsr_pipeline_example_test_rsr.json`
  - tiny RSR dry-run config

- `configs/rsr_pipeline.example.json`
  - backward-compatible alias of the GRACE example config

- `requirements.txt`
  - minimal Python dependencies for this root-level pipeline

## Single Run

```bash
python teacher_rsr_selection.py \
  --teacher-folder dataset/gsm8k_teacher_output \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --model-name qwen2.5-0.5b-instruct \
  --selection-metric grace \
  --sample-size 200 \
  --output-root outputs/gsm8k_qwen05b
```

This creates:

```text
outputs/gsm8k_qwen05b/
  final_dataset/
    <selected>.jsonl
    dataset_info.json
  teacher_ranking.json
  teacher_ranking.tsv
  run_summary.json
  ...
```

## Batch Run

```bash
python rsr_pipeline.py \
  --config configs/rsr_pipeline.example_grace.json \
  --output-root outputs/pipeline_runs
```

or

```bash
bash run_rsr_pipeline.sh configs/rsr_pipeline.example_grace.json --output-root outputs/pipeline_runs
```

For RSR, use:

```bash
python rsr_pipeline.py \
  --config configs/rsr_pipeline.example_rsr.json \
  --output-root outputs/pipeline_runs
```

## Config Notes

- `defaults`
  - shared defaults for datasets/models/runs

- `models`
  - each item needs:
    - `name`
    - `model_path`

- `datasets`
  - each item needs:
    - `name`
    - `teacher_folder`
  - common optional fields:
    - `teacher_glob`
    - `dataset_info_path`
    - `dataset_export_dir`
    - `selection_metric`
    - `sample_size`
    - `seed`
    - `batch_size`
    - `dtype`
    - `chat_template`
    - `grace_projection_dim`
    - `grace_projection_seed`
    - `grace_projection_chunk_size`
    - `grace_num_partitions`
    - `grace_smoothing`
    - `max_model_len`
    - `gpus_per_worker`
    - `max_workers`

- `runs`
  - optional explicit matrix
  - each item references:
    - `dataset`
    - `model`
  - can override any per-run field

If `runs` is omitted, `rsr_pipeline.py` uses the full dataset × model cross product.

Supported `selection_metric` values are `rsr`, `grace`, `g_norm`, and `g_vendi`. For paper-style GRACE runs, use `batch_size=1`, `sample_size=200`, `grace_projection_dim=512`, and `grace_num_partitions=10`.

Recommended practice is to keep separate config files for `grace` and `rsr`, because the metric-specific parameters and export directories are different. This repo now includes:

- `configs/rsr_pipeline.example_grace.json`
- `configs/rsr_pipeline.example_rsr.json`
- `configs/rsr_pipeline_example_test_grace.json`
- `configs/rsr_pipeline_example_test_rsr.json`

The older files `configs/rsr_pipeline.example.json` and `configs/rsr_pipeline_example_test.json` remain available as GRACE aliases.

# RSR Pipeline

## Files

- `teacher_rsr_selection.py`
  - single run entrypoint
  - input: one teacher-output folder + one student model
  - output: ranking files + `final_dataset/` with selected jsonl and `dataset_info.json`

- `rsr_pipeline.py`
  - batch runner for multiple datasets and multiple student models
  - reads a JSON config

- `run_rsr_pipeline.sh`
  - thin shell wrapper around `rsr_pipeline.py`

- `configs/rsr_pipeline.example.json`
  - example batch config

- `requirements.txt`
  - minimal Python dependencies for this root-level pipeline

## Single Run

```bash
python teacher_rsr_selection.py \
  --teacher-folder dataset/gsm8k_teacher_output \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --model-name qwen2.5-0.5b-instruct \
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
  --config configs/rsr_pipeline.example.json \
  --output-root outputs/pipeline_runs
```

or

```bash
bash run_rsr_pipeline.sh configs/rsr_pipeline.example.json --output-root outputs/pipeline_runs
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
    - `sample_size`
    - `seed`
    - `batch_size`
    - `dtype`
    - `chat_template`
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

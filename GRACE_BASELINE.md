# GRACE Baseline in This Repo

This file explains exactly how the GRACE teacher-selection baseline is implemented in this repo, what its inputs and outputs are, which CLI parameters were added, and how it maps to:

1. the latest GRACE paper submission, `20641_In_Good_GRACES_Principle.pdf`
2. the simplified GRACE baseline used in the RSR paper, `baseline_1_RSR.pdf`, Appendix B.5

The goal is to make it unambiguous what score is computed, on what data, what gets exported after selection, and where this repo intentionally differs from the exact released GRACE experimental stack.

Primary references inside `20641_In_Good_GRACES_Principle.pdf`:

- Section 2.1
  - gradient definition, projection, `log(|y|)` rescaling, `Sigma`, `tilde(Sigma)`, `hat(Sigma)`

- Section 2.2
  - GRACE definition and cross-validation trace formula

- Section 3
  - main experimental defaults: `n=512`, `m=4`, `d=512`, `C=10`

- Appendix F.3 / F.4
  - hyperparameter ablations and the motivation for `log(T)` scaling

- Appendix G
  - student prompt format used in the original GRACE experiments

## 1. Short Version

For each candidate teacher file:

1. Sample the same shared subset of `sample_size` examples across all teachers.
2. For each sampled example, compute the student gradient on the teacher response.
3. Project that full gradient to a low-dimensional vector.
4. Multiply the projected vector by `log(response_length)`.
5. Split the sampled set into `C` folds.
6. Compute one dataset-level scalar `GRACE` score for that teacher.
7. Pick the teacher with the smallest `GRACE`.
8. Export the full original teacher file, not just the sampled subset.

So yes: GRACE here is also a "mini-score" for teacher selection.

- It is computed only on the sampled subset.
- It produces one scalar per teacher.
- The selected output is still the full teacher dataset.
- The pipeline still writes an output folder and `dataset_info.json`, same as the RSR pipeline.

## 2. What Was Added

The selector now supports:

- `rsr`
- `grace`
- `g_norm`
- `g_vendi`

The new metric is chosen with:

```bash
--selection-metric grace
```

## 3. New Parameters

These are the GRACE-related parameters added to `teacher_rsr_selection.py`.

| Parameter | Meaning | Used by |
| --- | --- | --- |
| `--selection-metric` | Which teacher-selection score to use: `rsr`, `grace`, `g_norm`, `g_vendi` | all |
| `--grace-projection-dim` | Low-dimensional projection size `d` for projected gradients | `grace`, `g_norm`, `g_vendi` |
| `--grace-projection-seed` | Seed for the fixed random projection matrix; `-1` means reuse `--seed` | `grace`, `g_norm`, `g_vendi` |
| `--grace-projection-chunk-size` | Chunk size used to stream the projection over large parameter vectors | `grace`, `g_norm`, `g_vendi` |
| `--grace-num-partitions` | Cross-validation fold count `C` in GRACE | `grace` |
| `--grace-smoothing` | Smoothing coefficient `nu` in `hat(Sigma) = tilde(Sigma) + nu/d * I` | `grace` |
| `--dataset-export-dir` | Where the selected full teacher file is copied; default becomes metric-specific | all |

The old RSR parameters still exist. They are ignored when the chosen metric does not need them.

Example:

```bash
python teacher_rsr_selection.py \
  --teacher-folder dataset/gsm8k_teacher_output \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --selection-metric grace \
  --sample-size 200 \
  --batch-size 1 \
  --grace-projection-dim 512 \
  --grace-num-partitions 10 \
  --grace-smoothing 1e-4
```

## 4. Input and Output of the GRACE Score

### 4.1 Raw Teacher Dataset Record

The selector starts from the raw teacher dataset file.

For example, one line in `dataset/gsm8k_teacher_output/gpt-4.1_2025-04-14_teacher_responses.jsonl` looks like:

```json
{
  "id": 0,
  "system": "You are a helpful assistant.\nPlease answer the math question below exactly as asked.",
  "instruction": "Mimi picked up 2 dozen seashells on the beach. Kyle found twice as many shells as Mimi and put them in his pocket. Leigh grabbed one-third of the shells that Kyle found. How many seashells did Leigh have?",
  "teacher_output": "Let's solve the problem step by step:\n\n1. Mimi picked up 2 dozen seashells...\n\nFinal Answer: Leigh had 16 seashells.",
  "output": "Mimi has 2 x 12 = <<2*12=24>>24 sea shells.\nKyle has 24 x 2 = <<24*2=48>>48 sea shells.\nLeigh has 48 / 3 = <<48/3=16>>16 sea shells.\n#### 16",
  "teacher_model": "gpt-4.1_2025-04-14"
}
```

For GRACE scoring in this repo, the fields that matter are:

- `id`
- `system`
- `instruction`
- `teacher_output`

Fields such as `output`, `teacher_model`, or `category` are preserved in the full copied teacher file, but they are not used when computing the GRACE mini-score.

### 4.2 Shared Sampling

For one teacher, the score input is a sampled dataset:

`D = {(x_i, y_i)} for i = 1..N_sampled`

In this repo:

- `x_i` is the prompt built from the sampled record
- `y_i` is the teacher response in `teacher_output`
- `N_sampled = sample_size` unless fewer common ids are available

The important point is that sampling is shared across teachers:

1. scan every teacher file
2. find ids that exist in all teachers
3. sample `sample_size` ids once
4. use exactly that same id subset for every teacher

That shared subset is written to:

```text
output_root/sampled_ids.json
```

### 4.3 Prepared Record Written by the Selector

Each sampled raw record is rewritten into a scoring-ready JSONL record under:

```text
output_root/prepared_messages/<teacher_name>.jsonl
```

Example prepared row:

```json
{
  "id": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant.\nPlease answer the math question below exactly as asked."
    },
    {
      "role": "user",
      "content": "Mimi picked up 2 dozen seashells on the beach. Kyle found twice as many shells as Mimi and put them in his pocket. Leigh grabbed one-third of the shells that Kyle found. How many seashells did Leigh have?"
    },
    {
      "role": "assistant",
      "content": "Let's solve the problem step by step:\n\n1. Mimi picked up 2 dozen seashells...\n\nFinal Answer: Leigh had 16 seashells."
    }
  ],
  "teacher_name": "gpt-4.1_2025-04-14",
  "source_path": "dataset/gsm8k_teacher_output/gpt-4.1_2025-04-14_teacher_responses.jsonl"
}
```

This prepared record is the actual input consumed by the student-side GRACE scorer.

### 4.4 Per-Sample Gradient Computation

For each prepared sample:

1. tokenize the `messages`
2. identify only the assistant-response tokens
3. compute the student's mean token cross-entropy on those response tokens
4. backpropagate once to get the full student gradient `g(x_i, y_i)`
5. project that full gradient to `R^d`
6. multiply by `log(response_length)`

So for each sample:

- input:
  - one prepared JSONL row
- intermediate outputs:
  - full gradient `g(x_i, y_i) in R^D`
  - projected gradient `Pi g(x_i, y_i) in R^d`
  - processed gradient `h(x_i, y_i) = log(|y_i|) * Pi g(x_i, y_i) in R^d`

If `sample_size = 200` and `grace_projection_dim = 512`, then the stacked processed gradient matrix has shape:

`G(D) in R^(200 x 512)`

### 4.5 Per-Sample Intermediate Output

For each sampled example, the GRACE path writes diagnostic rows to:

```text
output_root/sample_metrics/<teacher_name>.jsonl
```

Example row:

```json
{
  "sample_id": 0,
  "resp_token_length": 104,
  "avg_surprisal": 1.2345,
  "projected_grad_norm": 0.4567,
  "processed_grad_norm": 2.1198
}
```

Important:

- these are diagnostics only
- GRACE itself is not a per-sample score
- teacher ranking uses only the dataset-level scalar

### 4.6 Dataset-Level Output

For each teacher, the pipeline produces one row in `teacher_ranking.json` with:

- `grace`
- `grace_variance`
- `grace_bias`
- `g_norm`
- `g_vendi`
- `selection_score`
- `selection_metric`
- `selection_objective`

For `--selection-metric grace`:

- `selection_score_key = "grace"`
- `selection_objective = "min"`
- lower is better

## 5. Exact Score Definition Used Here

### 5.1 Student Gradient

For one sampled example `(x, y)`, we compute the student gradient on the teacher response:

`g(x, y) = grad_theta CE_mean(y | x)`

This matches the paper intent:

- the gradient is with respect to student parameters
- it is averaged over response tokens

Implementation detail:

- the code uses the mean token cross-entropy over assistant-response tokens
- this is equivalent to the paper's token-averaged gradient definition

### 5.2 Low-Dimensional Projection

The original GRACE paper writes a fixed random matrix:

`Pi in {+/- 1/sqrt(D)}^(d x D)`

and defines the processed gradient using `Pi g`.

In this repo, the implementation uses a fixed sparse CountSketch-style random projection instead of materializing a dense `d x D` matrix.

Reason:

- for modern LMs, `D` is too large for a dense projection matrix to be practical
- the sparse projection preserves the same role in the algorithm: a fixed random map from the full gradient to `R^d`

So the repo computes:

`projected_g(x, y) in R^d`

with fixed seed-controlled random hashing/sign assignment.

### 5.3 Length Rescaling

The processed gradient is:

`h(x, y) = log(|y|) * projected_g(x, y)`

where `|y|` is the assistant response token length.

This follows the GRACE paper and the RSR appendix description.

Why:

- longer sequences tend to have smaller average gradient norms
- the log-length rescaling reduces the metric's bias toward short responses

Small implementation guard:

- the code uses `log(max(|y|, 2))`
- this avoids collapsing a 1-token response to an all-zero processed gradient
- for reasoning-trajectory style data, this is usually irrelevant because responses are typically much longer than 1 token

### 5.4 Dataset Matrices

For a teacher with `N` sampled examples, stack the processed gradients into:

`G(D) in R^(N x d)`

Each row is one `h(x_i, y_i)`.

Then define:

- unnormalized second moment:
  - `Sigma(D) = (1/N) * G(D)^T G(D)`
- normalized second moment:
  - first normalize each row to unit norm
  - then compute `tilde(Sigma)(D)`
- regularized inverse-covariance proxy:
  - `hat(Sigma)(D) = tilde(Sigma)(D) + (nu/d) * I`

where:

- `d = grace_projection_dim`
- `nu = grace_smoothing`

### 5.5 Cross-Validation GRACE

Let the sampled dataset be partitioned into `C` folds:

`D_1, ..., D_C`

In the original GRACE formulation, partitions are defined over prompts.
In this repo's RSR-style setup, each sampled prompt has one response, so partitioning sampled examples is equivalent to partitioning prompts.

For each fold `i`:

1. hold out `D_i`
2. use the remaining data `D_-i` as background
3. compute:
   - `Sigma(D_i)` from unnormalized processed gradients
   - `hat(Sigma)(D_-i)` from normalized processed gradients
4. compute:
   - `trace(hat(Sigma)(D_-i)^(-1) * Sigma(D_i))`

The final GRACE score is:

`GRACE(D) = (1/C) * sum_i trace(hat(Sigma)(D_-i)^(-1) * Sigma(D_i))`

Lower is better.

This is exactly the dataset-level scoring logic used for teacher ranking in this repo.

## 6. Paper Alignment

### 6.1 Original GRACE Paper

The original paper setting is:

- `N` prompts
- `M` generations per prompt
- evaluate on a smaller subsampled dataset
- partition by prompt groups

The repo matches the algorithmic structure:

- student gradient on teacher generations
- low-dimensional random projection
- `log(|y|)` rescaling
- normalized spectrum for `hat(Sigma)`
- cross-validation trace score

### 6.2 RSR Paper Simplification

The RSR paper baseline uses a much simpler experimental setup:

- each teacher contributes one response per sampled prompt
- they use only 200 sampled reasoning trajectories per teacher
- GRACE is used only for teacher selection
- after selecting the teacher, they train on the full selected dataset

This repo matches that simplified usage:

- one sampled response per id per teacher
- shared sampling across all teachers
- default `sample_size = 200`
- default `grace_projection_dim = 512`
- default `grace_num_partitions = 10`
- teacher selection by lowest dataset-level `grace`
- export the full original teacher file after selection

### 6.3 One Important Engineering Difference

The only deliberate engineering deviation is the projection implementation:

- paper notation: dense random matrix
- repo implementation: fixed sparse streaming random projection

This change was made for tractability on large models.

### 6.4 Important Experimental Differences from the Latest GRACE Paper

Even though the score definition follows the latest paper, this repo is not an exact reproduction of the original GRACE experiments.

Main experimental differences relative to `20641_In_Good_GRACES_Principle.pdf`:

1. Student input format
   - latest GRACE paper Appendix G: uses a plain text format
   - `### Problem: {input}`
   - `### Solution: {output}`
   - this repo: uses the teacher dataset's `(system, user, assistant)` chat structure

2. Original paper data structure
   - latest GRACE paper Section 3: uses `n = 512` prompts and `m = 4` generations per prompt
   - this repo's current selector is built around one sampled response per shared id in each teacher file, which matches the RSR-style simplified use case better than the original `512 x 4` setup

3. Student model family
   - latest GRACE paper studies base students such as LLaMA-1B / LLaMA-3B and formats data accordingly
   - this repo is built to work cleanly with chat-template scoring for generic teacher-selection pipelines

So:

- the score mathematics here is aligned to the latest GRACE paper
- the surrounding experimental stack is aligned more closely to the RSR-style teacher-selection pipeline

### 6.5 Comparison to the Author Repo in `./GRACE`

The author-provided repo under `./GRACE` is not identical to the paper formula in every implementation detail.

Main differences between `./GRACE` and this repo:

1. Length rescaling
   - paper / this repo: multiply each projected gradient by `log(|y|)`
   - author repo: no explicit `log(|y|)` rescaling in `GRACE/GRACE/gradient_computation.py`

2. Smoothing
   - paper / this repo: `hat(Sigma) = tilde(Sigma) + (nu/d) * I`
   - author repo: smooth eigenvalues by interpolating them toward their mean in `GRACE/GRACE/GRACE_computation.py`

3. Split strategy
   - RSR-style paper baseline / this repo: disjoint folds, leave-one-fold-out
   - author repo: repeated random train/test splits controlled by `test_fraction` and `n_splits`

4. Projection backend
   - this repo: fixed sparse streaming random projection for tractability on large models
   - author repo: `trak` CUDA projector with Rademacher projection

5. Input formatting
   - this repo: uses the actual `(system, user, assistant)` chat structure from the teacher files
   - author repo: tokenizes into a fixed `"### Problem"` / `"### Solution"` format

Because of these differences, you should not expect the absolute GRACE numbers from this repo to exactly match the numbers from the author's standalone repo.

However:

- this repo is closer to the GRACE formula described in the paper and to the simplified RSR Appendix B.5 procedure
- the author repo is closer to the exact released implementation used by the GRACE authors

## 7. What "Mini Score" Means Here

Yes, GRACE here is also a small-sample teacher-selection score.

Concretely:

1. discover the ids shared by all teacher files
2. sample `sample_size` ids once, with a fixed seed
3. compute one scalar GRACE score per teacher on only those sampled ids
4. select the best teacher using that scalar
5. copy the teacher's full original file into the output dataset location

So the score is mini, but the exported training dataset is full.

This matches the RSR paper baseline logic.

## 8. Output Folder and `dataset_info`

Yes, the GRACE path keeps the same output pattern as the RSR path.

For `--selection-metric grace`, the default output locations become metric-specific:

- run folder default:
  - `dataset/grace_teacher_selection/...`
- selected dataset export default:
  - `dataset/grace_selected_teachers/...`

Inside `output_root`, you still get:

```text
output_root/
  final_dataset/
    grace_selected__...jsonl
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

And yes:

- the winning full teacher file is copied out
- `final_dataset/dataset_info.json` is produced
- the global `dataset_info_path` is updated too

### 8.1 Concrete Output Example

Suppose:

- student model: `qwen2.5-0.5b-instruct`
- metric: `grace`
- sample size: `200`
- winning teacher: `gpt-4.1_2025-04-14`

Then the selected dataset entry name will be:

```text
grace_selected__qwen2.5-0.5b-instruct__n200__gpt-4.1_2025-04-14
```

The final copied dataset files will be:

```text
dataset/grace_selected_teachers/grace_selected__qwen2.5-0.5b-instruct__n200__gpt-4.1_2025-04-14.jsonl
output_root/final_dataset/grace_selected__qwen2.5-0.5b-instruct__n200__gpt-4.1_2025-04-14.jsonl
```

And `output_root/final_dataset/dataset_info.json` will look like:

```json
{
  "grace_selected__qwen2.5-0.5b-instruct__n200__gpt-4.1_2025-04-14": {
    "file_name": "grace_selected__qwen2.5-0.5b-instruct__n200__gpt-4.1_2025-04-14.jsonl",
    "columns": {
      "system": "system",
      "prompt": "instruction",
      "response": "teacher_output"
    }
  }
}
```

The global `dataset_info_path` is also updated with the same entry name, except its `file_name` becomes relative to the shared dataset root, for example:

```json
{
  "grace_selected__qwen2.5-0.5b-instruct__n200__gpt-4.1_2025-04-14": {
    "file_name": "grace_selected_teachers/grace_selected__qwen2.5-0.5b-instruct__n200__gpt-4.1_2025-04-14.jsonl",
    "columns": {
      "system": "system",
      "prompt": "instruction",
      "response": "teacher_output"
    }
  }
}
```

## 9. What Each Output File Means

| File | Meaning |
| --- | --- |
| `sampled_ids.json` | the shared subset used to compute the mini-score |
| `prepared_messages/*.jsonl` | normalized chat-format inputs used for scoring |
| `sample_metrics/*.jsonl` | per-sample diagnostics; not the selection score |
| `teacher_ranking.json` | teacher-level ranking with all dataset-level scores |
| `teacher_ranking.tsv` | tabular version of the ranking |
| `run_summary.json` | metadata, selected teacher, metric settings, output paths |
| `final_dataset/<name>.jsonl` | full copied teacher file that won selection |
| `final_dataset/dataset_info.json` | dataset registry for the selected output |

## 10. Selection Direction

Teacher choice depends on the metric:

| Metric | Score used | Better direction |
| --- | --- | --- |
| `rsr` | `rank_surprisal_ratio` | lower |
| `grace` | `grace` | lower |
| `g_norm` | `g_norm` | lower |
| `g_vendi` | `g_vendi` | higher |

## 11. Recommended Config Files

Yes, it is better to keep separate pipeline config files for `grace` and `rsr`.

Reasons:

1. `grace` has metric-specific parameters such as projection dimension, smoothing, and fold count
2. `rsr` has different metric-specific parameters such as `rank_clip_r`
3. output folders and dataset entry prefixes should stay separate
4. it avoids accidentally reusing a `grace` export directory for an `rsr` run, or vice versa

Recommended files in this repo:

- `configs/rsr_pipeline.example_grace.json`
- `configs/rsr_pipeline.example_rsr.json`
- `configs/rsr_pipeline_example_test_grace.json`
- `configs/rsr_pipeline_example_test_rsr.json`

Backward-compatible GRACE aliases are also kept:

- `configs/rsr_pipeline.example.json`
- `configs/rsr_pipeline_example_test.json`

## 12. Current Repo Defaults for GRACE

The example config now uses:

```json
{
  "selection_metric": "grace",
  "sample_size": 200,
  "batch_size": 1,
  "grace_projection_dim": 512,
  "grace_num_partitions": 10,
  "grace_smoothing": 0.0001
}
```

These match the simplified RSR-style baseline setup you described.

## 13. Implementation Notes and Guard Rails

1. `batch_size` must be `1` for `grace`, `g_norm`, and `g_vendi`.
   The code enforces this because gradients are extracted one sample at a time.

2. At least 2 shared samples are required.
   GRACE needs a non-empty held-out split and background split.

3. The projection is fixed across all teachers in a run.
   The same seed means teachers are compared in the same projected space.

4. GRACE is dataset-level only.
   The pipeline stores sample diagnostics, but selection uses the teacher-level scalar only.

## 14. Files That Implement GRACE

- `rsr_core.py`
  - `extract_processed_gradients`
  - `compute_gradient_baseline_metrics`
  - `FixedSparseRandomProjector`

- `teacher_rsr_selection.py`
  - CLI parameter definitions
  - metric routing
  - ranking
  - output export

- `rsr_pipeline.py`
  - config-to-CLI plumbing for the GRACE parameters

## 15. Practical Summary

If you run either of these:

```bash
python teacher_rsr_selection.py \
  --teacher-folder <teacher_dir> \
  --model-path <student_model> \
  --selection-metric grace \
  --sample-size 200 \
  --batch-size 1 \
  --grace-projection-dim 512 \
  --grace-num-partitions 10
```

or

```bash
python rsr_pipeline.py \
  --config configs/rsr_pipeline.example_grace.json \
  --dry-run
```

then the pipeline will:

1. sample 200 shared ids
2. compute one processed gradient vector per sampled trajectory
3. compute one scalar `grace` per teacher
4. choose the teacher with the minimum `grace`
5. copy that teacher's full original dataset to the output folder
6. generate `dataset_info.json` exactly like the RSR path

That is the GRACE baseline behavior in this repo.

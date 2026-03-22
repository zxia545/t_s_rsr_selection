#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
SELECTOR_SCRIPT = ROOT / "teacher_rsr_selection.py"


def load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("._") or "item"


def merge_options(shared: Dict, specific: Dict) -> Dict:
    merged = dict(shared)
    merged.update(specific)
    return merged


def normalize_models(config: Dict) -> Dict[str, Dict]:
    models = config.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError("config must include a non-empty `models` list")

    normalized = {}
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("each `models` entry must be an object")
        if not model.get("name") or not model.get("model_path"):
            raise ValueError("each model needs `name` and `model_path`")
        if model["name"] in normalized:
            raise ValueError(f"duplicate model name: {model['name']}")
        normalized[model["name"]] = model
    return normalized


def normalize_datasets(config: Dict) -> Dict[str, Dict]:
    datasets = config.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config must include a non-empty `datasets` list")

    normalized = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("each `datasets` entry must be an object")
        if not dataset.get("name") or not dataset.get("teacher_folder"):
            raise ValueError("each dataset needs `name` and `teacher_folder`")
        if dataset["name"] in normalized:
            raise ValueError(f"duplicate dataset name: {dataset['name']}")
        normalized[dataset["name"]] = dataset
    return normalized


def select_names(all_names: Sequence[str], include_arg: str) -> List[str]:
    if not include_arg:
        return list(all_names)
    wanted = [part.strip() for part in include_arg.split(",") if part.strip()]
    unknown = sorted(set(wanted) - set(all_names))
    if unknown:
        raise ValueError(f"unknown names requested: {', '.join(unknown)}")
    return wanted


def iter_runs(config: Dict, selected_model_names: Sequence[str], selected_dataset_names: Sequence[str]) -> Iterable[Tuple[Dict, Dict, Dict]]:
    shared_defaults = config.get("defaults", {})
    run_matrix = config.get("runs")
    models = normalize_models(config)
    datasets = normalize_datasets(config)

    if run_matrix is None:
        for dataset_name in selected_dataset_names:
            for model_name in selected_model_names:
                yield (
                    merge_options(shared_defaults, datasets[dataset_name]),
                    merge_options(shared_defaults, models[model_name]),
                    {},
                )
        return

    if not isinstance(run_matrix, list):
        raise ValueError("`runs` must be a list when provided")

    for run in run_matrix:
        if not isinstance(run, dict):
            raise ValueError("each `runs` entry must be an object")
        dataset_name = run.get("dataset")
        model_name = run.get("model")
        if dataset_name not in datasets:
            raise ValueError(f"run references unknown dataset: {dataset_name}")
        if model_name not in models:
            raise ValueError(f"run references unknown model: {model_name}")
        if dataset_name not in selected_dataset_names or model_name not in selected_model_names:
            continue
        yield (
            merge_options(shared_defaults, datasets[dataset_name]),
            merge_options(shared_defaults, models[model_name]),
            run,
        )


def resolve_path(value: Optional[str], base: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def build_command(dataset_cfg: Dict, model_cfg: Dict, run_cfg: Dict, global_output_root: Path) -> Tuple[List[str], str, Path]:
    dataset_name = dataset_cfg["name"]
    model_name = model_cfg["name"]
    selection_metric = (
        run_cfg.get("selection_metric")
        or dataset_cfg.get("selection_metric")
        or model_cfg.get("selection_metric")
        or "rsr"
    )

    run_name = run_cfg.get("name") or f"{sanitize_name(model_name)}__{sanitize_name(dataset_name)}"
    output_root = run_cfg.get("output_root") or dataset_cfg.get("output_root")
    if output_root:
        output_root_path = Path(output_root)
        if not output_root_path.is_absolute():
            output_root_path = (ROOT / output_root_path).resolve()
    else:
        output_root_path = (global_output_root / run_name).resolve()

    dataset_entry_name = run_cfg.get("dataset_entry_name")
    if not dataset_entry_name:
        dataset_entry_prefix = dataset_cfg.get("dataset_entry_prefix", f"{selection_metric}_selected")
        if selection_metric == "rsr_item":
            dataset_entry_name = (
                f"{dataset_entry_prefix}__{sanitize_name(model_name)}__all__{sanitize_name(dataset_name)}"
            )
        else:
            dataset_entry_name = (
                f"{dataset_entry_prefix}__{sanitize_name(model_name)}__n{run_cfg.get('sample_size', dataset_cfg.get('sample_size', 200))}"
                f"__{sanitize_name(dataset_name)}"
            )

    args = [
        sys.executable,
        str(SELECTOR_SCRIPT),
        "--teacher-folder",
        resolve_path(dataset_cfg["teacher_folder"], ROOT),
        "--model-path",
        model_cfg["model_path"],
        "--model-name",
        model_name,
        "--output-root",
        str(output_root_path),
        "--dataset-entry-name",
        dataset_entry_name,
    ]

    option_map = [
        ("teacher_glob", "--teacher-glob"),
        ("dataset_info_path", "--dataset-info-path"),
        ("dataset_export_dir", "--dataset-export-dir"),
        ("id_field", "--id-field"),
        ("system_field", "--system-field"),
        ("prompt_field", "--prompt-field"),
        ("response_field", "--response-field"),
        ("selection_metric", "--selection-metric"),
        ("sample_size", "--sample-size"),
        ("min_teachers_per_item", "--min-teachers-per-item"),
        ("seed", "--seed"),
        ("batch_size", "--batch-size"),
        ("dtype", "--dtype"),
        ("chat_template", "--chat-template"),
        ("rank_clip_r", "--rank-clip-r"),
        ("grace_projection_dim", "--grace-projection-dim"),
        ("grace_projection_seed", "--grace-projection-seed"),
        ("grace_projection_chunk_size", "--grace-projection-chunk-size"),
        ("grace_num_partitions", "--grace-num-partitions"),
        ("grace_smoothing", "--grace-smoothing"),
        ("max_model_len", "--max-model-len"),
        ("gpus_per_worker", "--gpus-per-worker"),
        ("max_workers", "--max-workers"),
    ]

    merged = {}
    merged.update(dataset_cfg)
    merged.update(model_cfg)
    merged.update(run_cfg)

    for key, cli_name in option_map:
        value = merged.get(key)
        if value is None or value == "":
            continue
        if key.endswith("_path") or key.endswith("_dir"):
            value = resolve_path(str(value), ROOT)
        args.extend([cli_name, str(value)])

    if merged.get("use_flash_attn"):
        args.append("--use-flash-attn")
    if merged.get("copy_selected_to_output", True):
        args.append("--copy-selected-to-output")

    return args, run_name, output_root_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch runner for teacher RSR selection experiments.")
    parser.add_argument("--config", required=True, help="JSON config describing datasets, models, and optional runs.")
    parser.add_argument("--output-root", default="pipeline_runs", help="Default parent output folder for runs.")
    parser.add_argument("--only-models", default="", help="Comma-separated subset of model names to run.")
    parser.add_argument("--only-datasets", default="", help="Comma-separated subset of dataset names to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on the first failed run.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)

    models = normalize_models(config)
    datasets = normalize_datasets(config)
    selected_model_names = select_names(list(models), args.only_models)
    selected_dataset_names = select_names(list(datasets), args.only_datasets)
    global_output_root = Path(args.output_root)
    if not global_output_root.is_absolute():
        global_output_root = (ROOT / global_output_root).resolve()
    global_output_root.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    failures = []

    for dataset_cfg, model_cfg, run_cfg in iter_runs(config, selected_model_names, selected_dataset_names):
        command, run_name, output_root_path = build_command(dataset_cfg, model_cfg, run_cfg, global_output_root)
        print(f"[pipeline] run={run_name}")
        print("[pipeline] cmd:", " ".join(command))

        final_dataset_dir = output_root_path / "final_dataset"
        if final_dataset_dir.exists():
            print(f"[pipeline] skip: found existing {final_dataset_dir}")
            run_summaries.append(
                {
                    "run_name": run_name,
                    "dataset": dataset_cfg["name"],
                    "model": model_cfg["name"],
                    "status": "skipped",
                    "skip_reason": f"existing final_dataset at {final_dataset_dir}",
                    "returncode": 0,
                }
            )
            continue

        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=str(ROOT))
        run_summary = {
            "run_name": run_name,
            "dataset": dataset_cfg["name"],
            "model": model_cfg["name"],
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
        }
        run_summaries.append(run_summary)

        if completed.returncode != 0:
            failures.append(run_summary)
            if args.fail_fast:
                break

    summary_path = global_output_root / "pipeline_summary.json"
    summary_payload = {
        "config": str(config_path),
        "selected_models": selected_model_names,
        "selected_datasets": selected_dataset_names,
        "dry_run": args.dry_run,
        "runs": run_summaries,
        "failures": failures,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[pipeline] summary: {summary_path}")

    if failures and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

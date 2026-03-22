#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from rsr_core import (
    compute_gradient_baseline_metrics,
    compute_sample_metrics,
    extract_processed_gradients,
    infer_dataset,
    load_model_and_tokenizer,
)


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("._") or "item"


def derive_model_name(model_name: str, model_path: str) -> str:
    if model_name:
        return model_name
    return Path(model_path.rstrip("/")).name or "student_model"


def derive_teacher_name(path: Path) -> str:
    name = path.stem
    for suffix in ("_teacher_responses", "_responses"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def json_default_sort_key(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def iter_records(path: Path) -> Iterable[Dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: invalid JSON on line {line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: line {line_no} is not a JSON object")
                yield record
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"{path}: element {idx} is not a JSON object")
            yield record
        return
    raise ValueError(f"{path}: expected JSON array or JSONL objects")


def validate_record(record: Dict, id_field: str, prompt_field: str, response_field: str) -> Tuple[bool, Optional[str]]:
    sample_id = record.get(id_field)
    if sample_id is None:
        return False, None

    prompt_text = normalize_text(record.get(prompt_field)).strip()
    response_text = normalize_text(record.get(response_field)).strip()
    if not prompt_text or not response_text:
        return False, None
    return True, sample_id


def collect_valid_ids(path: Path, id_field: str, prompt_field: str, response_field: str) -> Tuple[set, int]:
    valid_ids = set()
    duplicates = set()
    scanned = 0
    for record in iter_records(path):
        scanned += 1
        is_valid, sample_id = validate_record(record, id_field=id_field, prompt_field=prompt_field, response_field=response_field)
        if not is_valid:
            continue
        if sample_id in valid_ids:
            duplicates.add(sample_id)
        valid_ids.add(sample_id)
    if duplicates:
        dup_preview = ", ".join(json.dumps(x, ensure_ascii=False) for x in sorted(duplicates, key=json_default_sort_key)[:10])
        raise ValueError(f"{path}: duplicated `{id_field}` values found: {dup_preview}")
    return valid_ids, scanned


def collect_teacher_overview_and_common_ids(
    teacher_files: Sequence[Path],
    id_field: str,
    prompt_field: str,
    response_field: str,
) -> Tuple[List[Dict[str, object]], set]:
    common_ids: Optional[set] = None
    teacher_overview: List[Dict[str, object]] = []

    for path in teacher_files:
        teacher_name = derive_teacher_name(path)
        valid_ids, scanned_count = collect_valid_ids(
            path=path,
            id_field=id_field,
            prompt_field=prompt_field,
            response_field=response_field,
        )
        if common_ids is None:
            common_ids = set(valid_ids)
        else:
            common_ids &= valid_ids
        teacher_overview.append(
            {
                "teacher_name": teacher_name,
                "source_path": str(path),
                "scanned_count": scanned_count,
                "valid_count": len(valid_ids),
            }
        )

    common_valid_ids = common_ids or set()
    for row in teacher_overview:
        row["common_valid_id_count"] = len(common_valid_ids)
    return teacher_overview, common_valid_ids


def build_messages(system_text: str, prompt_text: str, response_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt_text})
    messages.append({"role": "assistant", "content": response_text})
    return messages


def prepare_teacher_job(
    source_path: Path,
    prepared_path: Path,
    teacher_name: str,
    sampled_ids: Sequence,
    sampled_id_set: set,
    id_field: str,
    system_field: str,
    prompt_field: str,
    response_field: str,
) -> Dict[str, object]:
    selected_records: Dict[object, Dict] = {}
    for record in iter_records(source_path):
        is_valid, sample_id = validate_record(
            record,
            id_field=id_field,
            prompt_field=prompt_field,
            response_field=response_field,
        )
        if not is_valid or sample_id not in sampled_id_set:
            continue
        selected_records[sample_id] = record

    missing_ids = [sid for sid in sampled_ids if sid not in selected_records]
    if missing_ids:
        preview = ", ".join(json.dumps(x, ensure_ascii=False) for x in missing_ids[:10])
        raise ValueError(f"{source_path}: missing sampled ids after filtering: {preview}")

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    with prepared_path.open("w", encoding="utf-8") as handle:
        for sample_id in sampled_ids:
            record = selected_records[sample_id]
            system_text = normalize_text(record.get(system_field))
            prompt_text = normalize_text(record.get(prompt_field))
            response_text = normalize_text(record.get(response_field))
            payload = {
                "id": sample_id,
                "messages": build_messages(system_text, prompt_text, response_text),
                "teacher_name": teacher_name,
                "source_path": str(source_path),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return {
        "teacher_name": teacher_name,
        "source_path": str(source_path),
        "prepared_path": str(prepared_path),
        "sample_count": len(sampled_ids),
    }


def detect_visible_gpu_ids() -> List[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None and cuda_visible_devices.strip():
        return [part.strip() for part in cuda_visible_devices.split(",") if part.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def build_device_groups(visible_gpu_ids: Sequence[str], gpus_per_worker: int, max_workers: int) -> List[List[str]]:
    if not visible_gpu_ids:
        return [[]]

    if gpus_per_worker <= 0:
        raise ValueError("--gpus-per-worker must be >= 1")

    if len(visible_gpu_ids) < gpus_per_worker:
        groups = [list(visible_gpu_ids)]
    else:
        full_group_count = len(visible_gpu_ids) // gpus_per_worker
        groups = [
            list(visible_gpu_ids[idx * gpus_per_worker : (idx + 1) * gpus_per_worker])
            for idx in range(full_group_count)
        ]

    if max_workers > 0:
        groups = groups[:max_workers]
    return groups or [[]]


def partition_jobs_round_robin(jobs: Sequence[Dict[str, object]], worker_count: int) -> List[List[Dict[str, object]]]:
    buckets: List[List[Dict[str, object]]] = [[] for _ in range(worker_count)]
    for idx, job in enumerate(jobs):
        buckets[idx % worker_count].append(job)
    return buckets


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(headers) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(header, "")) for header in headers) + "\n")


SELECTION_SCORE_KEYS = {
    "rsr": "rank_surprisal_ratio",
    "grace": "grace",
    "g_norm": "g_norm",
    "g_vendi": "g_vendi",
}

SELECTION_LABELS = {
    "rsr": "RSR",
    "grace": "GRACE",
    "g_norm": "G-Norm",
    "g_vendi": "G-Vendi",
}

DESCENDING_SELECTION_METRICS = {"g_vendi"}


def uses_gradient_baseline(selection_metric: str) -> bool:
    return selection_metric in {"grace", "g_norm", "g_vendi"}


def selection_score_key(selection_metric: str) -> str:
    return SELECTION_SCORE_KEYS[selection_metric]


def selection_label(selection_metric: str) -> str:
    return SELECTION_LABELS[selection_metric]


def selection_objective(selection_metric: str) -> str:
    return "max" if selection_metric in DESCENDING_SELECTION_METRICS else "min"


def load_dataset_info(dataset_info_path: Path) -> Dict[str, Dict]:
    if not dataset_info_path.exists():
        return {}
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_info, dict):
        raise ValueError(f"{dataset_info_path} must contain a JSON object")
    return dataset_info


def update_dataset_info(
    dataset_info_path: Path,
    dataset_entry_name: str,
    file_name_relative_to_dataset_root: str,
    system_field: str,
    prompt_field: str,
    response_field: str,
) -> Dict[str, Dict]:
    dataset_info = load_dataset_info(dataset_info_path)
    dataset_info[dataset_entry_name] = {
        "file_name": file_name_relative_to_dataset_root,
        "columns": {
            "system": system_field,
            "prompt": prompt_field,
            "response": response_field,
        },
    }

    ordered = {key: dataset_info[key] for key in sorted(dataset_info)}
    write_json(dataset_info_path, ordered)
    return ordered


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_existing_sampled_ids(sampled_ids_path: Path, common_ids: set) -> Optional[List[object]]:
    if not sampled_ids_path.exists():
        return None
    payload = json.loads(sampled_ids_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{sampled_ids_path} must contain a JSON array")
    missing_ids = [sid for sid in payload if sid not in common_ids]
    if missing_ids:
        preview = ", ".join(json.dumps(x, ensure_ascii=False) for x in missing_ids[:10])
        raise ValueError(f"{sampled_ids_path} contains ids that are no longer common across teachers: {preview}")
    return payload


def load_worker_result_rows(result_path: Path) -> Optional[List[Dict]]:
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{result_path} must contain a JSON array")
    return payload


def has_complete_sample_metrics(sample_metrics_path: Path, expected_count: int) -> bool:
    if not sample_metrics_path.exists():
        return False
    row_count = 0
    with sample_metrics_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{sample_metrics_path}: invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{sample_metrics_path}: line {line_no} is not a JSON object")
            row_count += 1
    return row_count == expected_count


def run_worker(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    worker_index = manifest["worker_index"]
    output_root = Path(manifest["output_root"])
    selection_metric = manifest["selection_metric"]
    score_key = selection_score_key(selection_metric)
    metric_label = selection_label(selection_metric)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[worker {worker_index}] starting with {len(jobs)} teacher files on "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '') or '<cpu>'}"
    )
    if uses_gradient_baseline(selection_metric) and manifest["batch_size"] != 1:
        raise ValueError(f"{metric_label} requires --batch-size 1 because gradients are computed per sample")

    result_path = Path(manifest["result_path"])
    existing_result_rows = load_worker_result_rows(result_path) or []
    existing_result_by_teacher = {row["teacher_name"]: row for row in existing_result_rows}

    completed_teacher_names = set()
    for job in jobs:
        teacher_name = job["teacher_name"]
        sample_metrics_path = output_root / "sample_metrics" / f"{sanitize_name(teacher_name)}.jsonl"
        existing_row = existing_result_by_teacher.get(teacher_name)
        if existing_row is None:
            continue
        if has_complete_sample_metrics(sample_metrics_path, int(job["sample_count"])):
            completed_teacher_names.add(teacher_name)

    if len(completed_teacher_names) == len(jobs):
        print(f"[worker {worker_index}] all {len(jobs)} teachers already completed; skipping worker")
        ordered_rows = [existing_result_by_teacher[job["teacher_name"]] for job in jobs]
        write_json(result_path, ordered_rows)
        return

    model, tokenizer = load_model_and_tokenizer(
        model_path=manifest["model_path"],
        dtype=manifest["dtype"],
        use_flash_attn=manifest["use_flash_attn"],
        chat_template=manifest["chat_template"],
        device_map=None if uses_gradient_baseline(selection_metric) else ("auto" if torch.cuda.is_available() else None),
    )

    results = []
    for job in jobs:
        teacher_name = job["teacher_name"]
        prepared_path = Path(job["prepared_path"])
        sample_metrics_path = output_root / "sample_metrics" / f"{sanitize_name(teacher_name)}.jsonl"
        if teacher_name in completed_teacher_names:
            print(f"[worker {worker_index}] resuming: skipping completed teacher {teacher_name}")
            results.append(existing_result_by_teacher[teacher_name])
            write_json(result_path, results)
            continue

        print(f"[worker {worker_index}] scoring {teacher_name} with {metric_label}")
        t0 = time.perf_counter()
        if uses_gradient_baseline(selection_metric):
            processed_gradients, sample_metrics = extract_processed_gradients(
                model=model,
                tokenizer=tokenizer,
                json_path=prepared_path,
                max_model_len=manifest["max_model_len"],
                chat_template=manifest["chat_template"],
                projection_dim=manifest["grace_projection_dim"],
                projection_seed=manifest["grace_projection_seed"],
                projection_chunk_size=manifest["grace_projection_chunk_size"],
            )
            infer_seconds = time.perf_counter() - t0
            dataset_metrics = compute_gradient_baseline_metrics(
                processed_gradients=processed_gradients,
                response_lengths=[row["resp_token_length"] for row in sample_metrics],
                num_partitions=manifest["grace_num_partitions"],
                smoothing=manifest["grace_smoothing"],
                partition_seed=manifest["seed"],
            )
        else:
            inferred_samples = infer_dataset(
                model=model,
                tokenizer=tokenizer,
                json_path=prepared_path,
                batch_size=manifest["batch_size"],
                max_model_len=manifest["max_model_len"],
                rank_clip_r=manifest["rank_clip_r"],
                chat_template=manifest["chat_template"],
            )
            infer_seconds = time.perf_counter() - t0
            dataset_metrics, sample_metrics = compute_sample_metrics(
                inferred_samples=inferred_samples,
                rank_clip_r=manifest["rank_clip_r"],
            )
        write_jsonl(sample_metrics_path, sample_metrics)

        if not dataset_metrics:
            raise RuntimeError(f"{teacher_name}: no valid {metric_label} metrics were produced")

        result_row = {
            "teacher_name": teacher_name,
            "source_path": job["source_path"],
            "prepared_path": job["prepared_path"],
            "sample_metrics_path": str(sample_metrics_path),
            "sample_count": job["sample_count"],
            "infer_seconds": round(infer_seconds, 6),
            "worker_index": worker_index,
            "selection_metric": selection_metric,
            "selection_score_key": score_key,
            "selection_objective": selection_objective(selection_metric),
            "selection_score": dataset_metrics[score_key],
            **dataset_metrics,
        }
        results.append(result_row)
        write_json(result_path, results)
    print(f"[worker {worker_index}] finished")


def build_controller_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a shared subset across teacher JSONL files, score each teacher with RSR or gradient baselines, and rank them."
    )
    parser.add_argument("--teacher-folder", required=True, help="Folder containing one teacher JSONL/JSON file per teacher.")
    parser.add_argument("--model-path", required=True, help="HF model name or local path for the student model.")
    parser.add_argument("--model-name", default="", help="Optional display name for the student model.")
    parser.add_argument("--output-root", default="", help="Folder for ranking outputs, prepared messages, logs, and manifests.")
    parser.add_argument(
        "--dataset-info-path",
        default="",
        help="Optional global dataset_info.json to update. Leave empty if you only need output_root/final_dataset.",
    )
    parser.add_argument(
        "--dataset-export-dir",
        default="",
        help="Where the winning full teacher file is copied. Defaults to dataset/<metric>_selected_teachers.",
    )
    parser.add_argument("--dataset-entry-name", default="", help="Optional dataset_info key for the selected teacher copy.")
    parser.add_argument("--teacher-glob", default="*.jsonl", help="Glob used to discover teacher files.")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--system-field", default="system")
    parser.add_argument("--prompt-field", default="instruction")
    parser.add_argument("--response-field", default="teacher_output")
    parser.add_argument(
        "--selection-metric",
        default="rsr",
        choices=["rsr", "grace", "g_norm", "g_vendi"],
        help="Teacher-selection score: token-level RSR, dataset-level GRACE, or GRACE paper baselines.",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--chat-template", default="qwen", choices=["qwen", "llama3"])
    parser.add_argument("--rank-clip-r", type=int, default=100)
    parser.add_argument(
        "--grace-projection-dim",
        type=int,
        default=512,
        help="Projected gradient dimension d used by GRACE / G-Norm / G-Vendi.",
    )
    parser.add_argument("--grace-projection-seed", type=int, default=-1, help="Defaults to --seed when set to -1.")
    parser.add_argument(
        "--grace-projection-chunk-size",
        type=int,
        default=1_048_576,
        help="Chunk size for the streaming sparse random projection over model parameters.",
    )
    parser.add_argument(
        "--grace-num-partitions",
        type=int,
        default=10,
        help="Cross-validation fold count C for GRACE. Effective value is capped by sample size.",
    )
    parser.add_argument(
        "--grace-smoothing",
        type=float,
        default=1e-4,
        help="Smoothing coefficient nu used in hat{Sigma}=tilde{Sigma}+nu/d*I.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=0, help="0 means auto from visible GPUs.")
    parser.add_argument("--copy-selected-to-output", action="store_true", help="Also copy the winning original teacher file into output_root.")
    parser.add_argument("--worker-manifest", default="", help=argparse.SUPPRESS)
    return parser


def run_controller(args: argparse.Namespace) -> None:
    model_name = derive_model_name(args.model_name, args.model_path)
    score_key = selection_score_key(args.selection_metric)
    metric_label = selection_label(args.selection_metric)
    grace_projection_seed = args.seed if args.grace_projection_seed < 0 else args.grace_projection_seed
    teacher_folder = Path(args.teacher_folder).resolve()
    if not teacher_folder.exists():
        raise FileNotFoundError(f"teacher folder does not exist: {teacher_folder}")

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = (
            Path(f"dataset/{args.selection_metric}_teacher_selection")
            / f"{sanitize_name(model_name)}__{sanitize_name(teacher_folder.name)}__n{args.sample_size}__seed{args.seed}"
        ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_info_path = Path(args.dataset_info_path).resolve() if args.dataset_info_path else None
    dataset_export_dir = Path(args.dataset_export_dir).resolve() if args.dataset_export_dir else None
    if dataset_export_dir is not None:
        dataset_export_dir.mkdir(parents=True, exist_ok=True)

    if uses_gradient_baseline(args.selection_metric) and args.batch_size != 1:
        raise ValueError(f"{metric_label} requires --batch-size 1 because gradients are computed per sample")

    teacher_files = sorted(teacher_folder.glob(args.teacher_glob))
    if not teacher_files:
        raise FileNotFoundError(f"no teacher files matched {args.teacher_glob!r} under {teacher_folder}")

    print(f"[controller] discovered {len(teacher_files)} teacher files")
    teacher_overview, common_ids = collect_teacher_overview_and_common_ids(
        teacher_files=teacher_files,
        id_field=args.id_field,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
    )
    if not common_ids:
        raise RuntimeError("no common valid sample ids were found across teachers")

    available_count = len(common_ids)
    sample_size = min(args.sample_size, available_count)
    if sample_size <= 0:
        raise ValueError("--sample-size must be >= 1")
    if uses_gradient_baseline(args.selection_metric) and sample_size < 2:
        raise ValueError(f"{metric_label} requires at least 2 shared samples")
    if sample_size < args.sample_size:
        print(
            f"[controller] requested sample_size={args.sample_size}, but only {available_count} common ids are available; "
            f"using {sample_size}"
        )

    sampled_ids_path = output_root / "sampled_ids.json"
    existing_sampled_ids = load_existing_sampled_ids(sampled_ids_path, common_ids)
    resumed_from_existing_sample = existing_sampled_ids is not None
    if existing_sampled_ids is not None:
        if len(existing_sampled_ids) != sample_size:
            raise ValueError(
                f"{sampled_ids_path} contains {len(existing_sampled_ids)} ids, expected {sample_size}. "
                "Delete the old run folder if you want to start over with a different sample size."
            )
        sampled_ids = existing_sampled_ids
        print(f"[controller] resuming with existing sampled ids from {sampled_ids_path}")
    else:
        rng = random.Random(args.seed)
        sorted_common_ids = sorted(common_ids, key=json_default_sort_key)
        sampled_ids = rng.sample(sorted_common_ids, k=sample_size)
        write_json(sampled_ids_path, sampled_ids)
    sampled_id_set = set(sampled_ids)
    write_json(output_root / "teacher_overview.json", teacher_overview)

    print(f"[controller] preparing sampled messages for {sample_size} shared ids")
    prepared_dir = output_root / "prepared_messages"
    jobs = []
    for path in teacher_files:
        teacher_name = derive_teacher_name(path)
        prepared_path = prepared_dir / f"{sanitize_name(teacher_name)}.jsonl"
        jobs.append(
            prepare_teacher_job(
                source_path=path,
                prepared_path=prepared_path,
                teacher_name=teacher_name,
                sampled_ids=sampled_ids,
                sampled_id_set=sampled_id_set,
                id_field=args.id_field,
                system_field=args.system_field,
                prompt_field=args.prompt_field,
                response_field=args.response_field,
            )
        )

    visible_gpu_ids = detect_visible_gpu_ids()
    device_groups = build_device_groups(
        visible_gpu_ids=visible_gpu_ids,
        gpus_per_worker=args.gpus_per_worker,
        max_workers=args.max_workers,
    )
    worker_job_groups = partition_jobs_round_robin(jobs, len(device_groups))

    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    worker_outputs_dir = output_root / "worker_outputs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_outputs_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    launched_workers = []
    result_paths: List[Path] = []
    resumed_worker_count = 0
    for worker_index, (device_group, worker_jobs) in enumerate(zip(device_groups, worker_job_groups)):
        if not worker_jobs:
            continue

        worker_name = f"worker_{worker_index:02d}"
        result_path = worker_outputs_dir / f"{worker_name}_results.json"
        manifest_path = manifests_dir / f"{worker_name}.json"
        manifest = {
            "worker_index": worker_index,
            "output_root": str(output_root),
            "result_path": str(result_path),
            "model_path": args.model_path,
            "selection_metric": args.selection_metric,
            "dtype": args.dtype,
            "chat_template": args.chat_template,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "rank_clip_r": args.rank_clip_r,
            "grace_projection_dim": args.grace_projection_dim,
            "grace_projection_seed": grace_projection_seed,
            "grace_projection_chunk_size": args.grace_projection_chunk_size,
            "grace_num_partitions": args.grace_num_partitions,
            "grace_smoothing": args.grace_smoothing,
            "seed": args.seed,
            "use_flash_attn": args.use_flash_attn,
            "jobs": worker_jobs,
        }
        write_json(manifest_path, manifest)
        result_paths.append(result_path)

        existing_rows = load_worker_result_rows(result_path)
        if existing_rows is not None:
            print(f"[controller] resuming: reusing {result_path}")
            resumed_worker_count += 1
            continue

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        if device_group:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(device_group)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""

        log_path = logs_dir / f"{worker_name}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        print(
            f"[controller] launching {worker_name} with {len(worker_jobs)} teachers on "
            f"{env['CUDA_VISIBLE_DEVICES'] or '<cpu>'}"
        )
        process = subprocess.Popen(
            [sys.executable, str(script_path), "--worker-manifest", str(manifest_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(script_path.parent),
            env=env,
        )
        launched_workers.append((worker_name, process, log_handle, log_path, result_path, device_group))

    for worker_name, process, log_handle, log_path, result_path, device_group in launched_workers:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            raise RuntimeError(
                f"{worker_name} failed with exit code {return_code}. "
                f"See log: {log_path} (devices: {','.join(device_group) or '<cpu>'})"
            )
        if not result_path.exists():
            raise RuntimeError(f"{worker_name} completed but did not produce {result_path}")

    if not result_paths:
        raise RuntimeError("no worker outputs were configured")

    ranking_rows: List[Dict] = []
    for result_path in result_paths:
        worker_rows = load_worker_result_rows(result_path)
        if worker_rows is None:
            raise RuntimeError(f"missing worker result: {result_path}")
        ranking_rows.extend(worker_rows)

    if not ranking_rows:
        raise RuntimeError("no teacher scores were produced")
    if args.selection_metric in DESCENDING_SELECTION_METRICS:
        ranking_rows.sort(key=lambda row: (-row["selection_score"], row["teacher_name"]))
    else:
        ranking_rows.sort(key=lambda row: (row["selection_score"], row["teacher_name"]))

    ranking_path = output_root / "teacher_ranking.json"
    ranking_tsv_path = output_root / "teacher_ranking.tsv"
    write_json(ranking_path, ranking_rows)
    write_tsv(ranking_tsv_path, ranking_rows)

    best_row = ranking_rows[0]
    best_teacher_name = best_row["teacher_name"]
    best_source_path = Path(best_row["source_path"])

    dataset_entry_name = args.dataset_entry_name or (
        f"{args.selection_metric}_selected__{sanitize_name(model_name)}__n{sample_size}__{sanitize_name(best_teacher_name)}"
    )

    output_final_dir = output_root / "final_dataset"
    output_final_dir.mkdir(parents=True, exist_ok=True)
    output_selected_copy_path = output_final_dir / f"{dataset_entry_name}.jsonl"
    shutil.copy2(best_source_path, output_selected_copy_path)
    output_dataset_info_path = output_final_dir / "dataset_info.json"

    selected_dataset_copy_path = None
    global_dataset_info = {}
    if dataset_export_dir is not None:
        selected_dataset_copy_path = dataset_export_dir / f"{dataset_entry_name}.jsonl"
        shutil.copy2(best_source_path, selected_dataset_copy_path)

    if dataset_info_path is not None:
        if selected_dataset_copy_path is None:
            raise ValueError("--dataset-info-path requires --dataset-export-dir so the exported dataset file can be indexed")
        dataset_root = dataset_info_path.parent
        try:
            relative_dataset_file = selected_dataset_copy_path.relative_to(dataset_root).as_posix()
        except ValueError:
            relative_dataset_file = os.path.relpath(selected_dataset_copy_path, dataset_root).replace(os.sep, "/")
        global_dataset_info = update_dataset_info(
            dataset_info_path=dataset_info_path,
            dataset_entry_name=dataset_entry_name,
            file_name_relative_to_dataset_root=relative_dataset_file,
            system_field=args.system_field,
            prompt_field=args.prompt_field,
            response_field=args.response_field,
        )

    output_dataset_info = {
        dataset_entry_name: {
            "file_name": output_selected_copy_path.name,
            "columns": {
                "system": args.system_field,
                "prompt": args.prompt_field,
                "response": args.response_field,
            },
        }
    }
    write_json(output_dataset_info_path, output_dataset_info)

    if args.copy_selected_to_output:
        selected_teacher_copy_path = output_root / "selected_teacher" / best_source_path.name
        selected_teacher_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_source_path, selected_teacher_copy_path)

    summary = {
        "created_at": iso_now(),
        "teacher_folder": str(teacher_folder),
        "teacher_glob": args.teacher_glob,
        "teacher_count": len(teacher_files),
        "common_valid_id_count": available_count,
        "model_path": args.model_path,
        "model_name": model_name,
        "selection_metric": args.selection_metric,
        "selection_score_key": score_key,
        "selection_objective": selection_objective(args.selection_metric),
        "dtype": args.dtype,
        "chat_template": args.chat_template,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "rank_clip_r": args.rank_clip_r,
        "grace_projection_dim": args.grace_projection_dim,
        "grace_projection_seed": grace_projection_seed,
        "grace_projection_chunk_size": args.grace_projection_chunk_size,
        "grace_num_partitions": args.grace_num_partitions,
        "grace_smoothing": args.grace_smoothing,
        "sample_size_requested": args.sample_size,
        "sample_size_used": sample_size,
        "resumed_from_existing_sample": resumed_from_existing_sample,
        "resumed_worker_count": resumed_worker_count,
        "seed": args.seed,
        "sampled_ids_path": str(sampled_ids_path),
        "ranking_path": str(ranking_path),
        "ranking_tsv_path": str(ranking_tsv_path),
        "dataset_info_path": str(dataset_info_path) if dataset_info_path is not None else "",
        "dataset_entry_name": dataset_entry_name,
        "selected_dataset_copy_path": str(selected_dataset_copy_path) if selected_dataset_copy_path is not None else "",
        "selected_output_copy_path": str(output_selected_copy_path),
        "output_dataset_info_path": str(output_dataset_info_path),
        "best_teacher": best_row,
        "visible_gpu_ids": visible_gpu_ids,
        "gpus_per_worker": args.gpus_per_worker,
        "worker_device_groups": device_groups,
        "global_dataset_info_entry": global_dataset_info.get(dataset_entry_name, {}),
    }
    write_json(output_root / "run_summary.json", summary)

    print(f"[controller] ranking saved to {ranking_path}")
    print(f"[controller] best teacher: {best_teacher_name} ({metric_label}={best_row[score_key]:.6f})")
    print(f"[controller] selected dataset copy: {output_selected_copy_path}")
    if selected_dataset_copy_path is not None:
        print(f"[controller] exported dataset copy: {selected_dataset_copy_path}")
    if dataset_info_path is not None:
        print(f"[controller] dataset_info updated: {dataset_info_path} -> {dataset_entry_name}")


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--worker-manifest", default="")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.worker_manifest:
        run_worker(Path(pre_args.worker_manifest).resolve())
        return

    parser = build_controller_arg_parser()
    args = parser.parse_args()
    run_controller(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ASSISTANT_MARKERS = {
    "qwen": ("<|im_start|>assistant\n", "<|im_end|>"),
    "llama3": ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"),
}

LLAMA3_FALLBACK_CHAT_TEMPLATE = (
    "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n"
    "{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n"
    "{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n"
    "{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n"
    "{#- This block extracts the system message, so we can slot it into the right place. #}\n"
    "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n"
    "    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n"
    "{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n"
    "{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n"
    "{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n"
    "{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n"
    "{%- if tools is not none and not tools_in_user_message %}\n"
    "    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n"
    "    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
    "    {{- \"Do not use variables.\\n\\n\" }}\n"
    "    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n"
    "{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n"
    "{#- Custom tools are passed in a user message with some extra guidance #}\n"
    "{%- if tools_in_user_message and not tools is none %}\n"
    "    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n"
    "        {%- set messages = messages[1:] %}\n    {%- else %}\n"
    "        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n"
    "    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n"
    "    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n"
    "    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n"
    "    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
    "    {{- \"Do not use variables.\\n\\n\" }}\n"
    "    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n"
    "    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n"
    "{%- for message in messages %}\n"
    "    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
    "        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n"
    "    {%- elif 'tool_calls' in message %}\n"
    "        {%- if not message.tool_calls|length == 1 %}\n"
    "            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n"
    "        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n"
    "        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n"
    "            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n"
    "            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n"
    "            {%- for arg_name, arg_val in tool_call.arguments | items %}\n"
    "                {{- arg_name + '=\"' + arg_val + '\"' }}\n"
    "                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n"
    "            {%- endfor %}\n            {{- \")\" }}\n"
    "        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n"
    "            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n"
    "            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n"
    "        {%- endif %}\n"
    "        {%- if builtin_tools is defined %}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n"
    "    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n"
    "        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n"
    "        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n"
    "        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
)


def _scan_response_positions(input_ids: List[int], header_ids: List[int], end_ids: List[int]) -> List[int]:
    positions = []
    n_tokens = len(input_ids)
    header_len = len(header_ids)
    end_len = len(end_ids)
    if header_len == 0:
        return positions

    index = 0
    while index <= n_tokens - header_len:
        if input_ids[index : index + header_len] == header_ids:
            start = index + header_len
            cursor = start
            found_end = False
            while cursor <= n_tokens - end_len:
                if end_len > 0 and input_ids[cursor : cursor + end_len] == end_ids:
                    found_end = True
                    break
                cursor += 1
            end = cursor if found_end else n_tokens
            if end > start:
                positions.extend(range(start, end))
            index = cursor + end_len if found_end else n_tokens
        else:
            index += 1
    return positions


def _process_messages(
    tokenizer,
    messages: List[Dict[str, str]],
    max_model_len: int,
    assistant_header_ids: List[int],
    assistant_end_ids: List[int],
) -> Tuple[List[int], List[int]]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if max_model_len:
        kwargs.update({"truncation": True, "max_length": max_model_len})

    encoded = tokenizer.apply_chat_template(messages, **kwargs)
    input_ids = encoded.get("input_ids", [])
    if torch.is_tensor(input_ids):
        input_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
    elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    if not input_ids:
        return [], []

    response_positions = _scan_response_positions(input_ids, assistant_header_ids, assistant_end_ids)
    return input_ids, response_positions


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "float16",
    use_flash_attn: bool = False,
    chat_template: str = "qwen",
    device_map: str = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    if not getattr(tokenizer, "chat_template", None):
        if chat_template == "llama3":
            tokenizer.chat_template = LLAMA3_FALLBACK_CHAT_TEMPLATE
        else:
            raise ValueError(
                "Tokenizer has no chat_template. "
                "Check the model tokenizer config or switch to --chat-template llama3 when appropriate."
            )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map[dtype]

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not device_map:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def infer_dataset(
    model,
    tokenizer,
    json_path: Path,
    batch_size: int,
    max_model_len: int = None,
    rank_clip_r: int = 100,
    chat_template: str = "qwen",
) -> List[Dict]:
    if json_path.suffix.lower() == ".jsonl":
        data = []
        with json_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        data = json.loads(json_path.read_text(encoding="utf-8"))

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pad_token_id = (
        text_tokenizer.pad_token_id
        if text_tokenizer.pad_token_id is not None
        else text_tokenizer.eos_token_id
    )
    assistant_header, assistant_end = ASSISTANT_MARKERS[chat_template]
    assistant_header_ids = text_tokenizer(assistant_header, add_special_tokens=False)["input_ids"]
    assistant_end_ids = text_tokenizer(assistant_end, add_special_tokens=False)["input_ids"]

    items = []
    for index, sample in enumerate(data):
        messages = sample.get("messages")
        if not messages:
            continue
        input_ids, response_positions = _process_messages(
            tokenizer=tokenizer,
            messages=messages,
            max_model_len=max_model_len,
            assistant_header_ids=assistant_header_ids,
            assistant_end_ids=assistant_end_ids,
        )
        if not input_ids or not response_positions:
            continue
        items.append(
            {
                "id": sample.get("id", index),
                "input_ids": input_ids,
                "response_positions": response_positions,
            }
        )

    if not items:
        return []

    inferred_samples: List[Dict] = []
    for start in tqdm(range(0, len(items), batch_size), desc="Inference", file=sys.stdout):
        batch = items[start : start + batch_size]
        input_ids_list = [item["input_ids"] for item in batch]
        max_len = max(len(ids) for ids in input_ids_list)

        input_ids_batch = []
        attention_mask_batch = []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            input_ids_batch.append(ids + [pad_token_id] * pad_len)
            attention_mask_batch.append([1] * len(ids) + [0] * pad_len)

        input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=model.device)
        attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=model.device)
        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            use_cache=False,
        )
        logits = outputs.logits

        for batch_index, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            response_positions = sorted(set(pos for pos in item["response_positions"] if 0 < pos < seq_len))
            logit_positions = [pos - 1 for pos in response_positions]
            target_ids = [item["input_ids"][pos] for pos in response_positions]
            if not logit_positions:
                continue

            valid_logits = logits[batch_index, logit_positions]
            targets = torch.tensor(target_ids, device=valid_logits.device, dtype=torch.long)
            logits_fp32 = valid_logits.float()
            log_partition = torch.logsumexp(logits_fp32, dim=-1)
            target_logits = logits_fp32.gather(1, targets[:, None]).squeeze(1)
            nll = log_partition - target_logits
            prob = torch.exp(-nll)

            top_values = torch.topk(
                valid_logits,
                k=min(rank_clip_r, valid_logits.shape[-1]),
                dim=-1,
            ).values
            rank = 1 + (top_values.float() > target_logits[:, None]).sum(dim=-1)
            rank = torch.clamp(rank, max=rank_clip_r)

            inferred_samples.append(
                {
                    "sample_id": item["id"],
                    "resp_token_positions": [int(pos) for pos in response_positions],
                    "probs": [float(value) for value in prob.detach().cpu().tolist()],
                    "nlls": [float(value) for value in nll.detach().cpu().tolist()],
                    "ranks": [int(value) for value in rank.detach().cpu().tolist()],
                }
            )

    return inferred_samples


def compute_sample_metrics(inferred_samples: List[Dict], rank_clip_r: int = 100) -> Tuple[Dict[str, float], List[Dict]]:
    if not inferred_samples:
        return {}, []

    eps = 1e-12
    sample_metrics = []
    for sample in inferred_samples:
        ranks = sample.get("ranks", [])
        nlls = sample.get("nlls", [])
        if not ranks or not nlls:
            continue
        if len(ranks) != len(nlls):
            raise ValueError(f"sample {sample['sample_id']} has mismatched ranks/nlls lengths")

        token_length = len(ranks)
        clipped_ranks = [min(rank, rank_clip_r) for rank in ranks]
        avg_rank = sum(clipped_ranks) / token_length
        avg_nll = sum(nlls) / token_length
        ratio = sum(clipped_ranks) / max(sum(nlls), eps)

        sample_metrics.append(
            {
                "sample_id": sample["sample_id"],
                "resp_token_length": token_length,
                "avg_rank_clip": avg_rank,
                "avg_surprisal": avg_nll,
                "rank_surprisal_ratio": ratio,
            }
        )

    if not sample_metrics:
        return {}, []

    count = len(sample_metrics)
    sum_avg_rank = sum(item["avg_rank_clip"] for item in sample_metrics)
    sum_avg_surprisal = sum(item["avg_surprisal"] for item in sample_metrics)
    sum_resp_token_length = sum(item["resp_token_length"] for item in sample_metrics)
    dataset_metrics = {
        "avg_resp_token_length": sum_resp_token_length / count,
        "avg_rank_clip": sum_avg_rank / count,
        "avg_surprisal": sum_avg_surprisal / count,
        "rank_surprisal_ratio": sum_avg_rank / max(sum_avg_surprisal, eps),
    }
    return dataset_metrics, sample_metrics

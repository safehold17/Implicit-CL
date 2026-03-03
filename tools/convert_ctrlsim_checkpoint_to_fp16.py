#!/usr/bin/env python3
"""Convert a Ctrl-Sim checkpoint from FP32 weights to FP16 weights."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ConvertStats:
    converted_tensors: int = 0
    converted_elements: int = 0
    fp16_tensors: int = 0
    other_tensors: int = 0


def _cast_fp32_to_fp16(obj: Any, stats: ConvertStats) -> Any:
    if torch.is_tensor(obj):
        if obj.dtype == torch.float32:
            stats.converted_tensors += 1
            stats.converted_elements += obj.numel()
            return obj.to(dtype=torch.float16)
        if obj.dtype == torch.float16:
            stats.fp16_tensors += 1
        else:
            stats.other_tensors += 1
        return obj

    if isinstance(obj, dict):
        return {key: _cast_fp32_to_fp16(value, stats) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_cast_fp32_to_fp16(value, stats) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_fp32_to_fp16(value, stats) for value in obj)

    return obj


def _is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    return all(torch.is_tensor(value) for value in obj.values())


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_fp16{input_path.suffix}")


def _convert_checkpoint_payload(
    checkpoint: Any, stats: ConvertStats, convert_all_fp32: bool
) -> tuple[Any, str]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        converted = dict(checkpoint)
        converted["state_dict"] = _cast_fp32_to_fp16(checkpoint["state_dict"], stats)
        target = "checkpoint['state_dict']"

        if convert_all_fp32:
            for key, value in checkpoint.items():
                if key == "state_dict":
                    continue
                converted[key] = _cast_fp32_to_fp16(value, stats)
            target = "entire checkpoint"

        return converted, target

    if _is_state_dict_like(checkpoint):
        return _cast_fp32_to_fp16(checkpoint, stats), "state_dict"

    raise ValueError(
        "Unsupported checkpoint format: expected either "
        "a Lightning checkpoint containing `state_dict`, or a plain state_dict."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ctrl-Sim checkpoint/model weights from FP32 to FP16."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input checkpoint (.ckpt/.pt/.pth).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output checkpoint. Defaults to <input_stem>_fp16<input_suffix>.",
    )
    parser.add_argument(
        "--drop-optimizer-states",
        action="store_true",
        help="Drop `optimizer_states` key after conversion to reduce file size.",
    )
    parser.add_argument(
        "--convert-all-fp32",
        action="store_true",
        help=(
            "Also convert FP32 tensors outside `state_dict` (for example optimizer "
            "states). Default only converts model weights."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(input_path)
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_path}. "
            "Use --overwrite to overwrite."
        )

    checkpoint = torch.load(input_path, map_location="cpu")
    stats = ConvertStats()

    converted_checkpoint, converted_target = _convert_checkpoint_payload(
        checkpoint=checkpoint,
        stats=stats,
        convert_all_fp32=args.convert_all_fp32,
    )

    if (
        args.drop_optimizer_states
        and isinstance(converted_checkpoint, dict)
        and "optimizer_states" in converted_checkpoint
    ):
        converted_checkpoint = dict(converted_checkpoint)
        converted_checkpoint.pop("optimizer_states", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_checkpoint, output_path)

    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Converted target: {converted_target}")
    print(f"Converted tensors (FP32 -> FP16): {stats.converted_tensors}")
    print(f"Converted elements: {stats.converted_elements}")
    print(f"Already FP16 tensors: {stats.fp16_tensors}")
    print(f"Other dtype tensors: {stats.other_tensors}")
    print(f"File size (MB): {input_size_mb:.2f} -> {output_size_mb:.2f}")


if __name__ == "__main__":
    main()

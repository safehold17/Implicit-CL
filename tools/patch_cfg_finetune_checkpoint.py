#!/usr/bin/env python3
"""Patch a raw finetune checkpoint with donor hyperparameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inject donor checkpoint hyperparameters into a raw finetune checkpoint."
    )
    parser.add_argument(
        "--finetune-input",
        required=True,
        help="Path to the raw finetune checkpoint.",
    )
    parser.add_argument(
        "--donor-input",
        required=True,
        help="Path to the donor checkpoint that already contains hyper_parameters.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the patched checkpoint. Defaults to <finetune_stem>_patched<suffix>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    """Build the default output path."""
    return input_path.with_name(f"{input_path.stem}_patched{input_path.suffix}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint payload from disk."""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict: {path}")
    return checkpoint


def patch_checkpoint(
    finetune_checkpoint: dict[str, Any],
    donor_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Inject donor hyperparameters into the finetune checkpoint."""
    if "hyper_parameters" not in donor_checkpoint:
        raise KeyError("Donor checkpoint does not contain `hyper_parameters`.")

    patched = dict(finetune_checkpoint)
    patched["hyper_parameters"] = donor_checkpoint["hyper_parameters"]

    if "hparams_name" in donor_checkpoint:
        patched["hparams_name"] = donor_checkpoint["hparams_name"]

    return patched


def main() -> None:
    """Patch the finetune checkpoint and write the result."""
    args = parse_args()

    finetune_input = Path(args.finetune_input).expanduser().resolve()
    donor_input = Path(args.donor_input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(finetune_input)
    )

    if not finetune_input.exists():
        raise FileNotFoundError(f"Finetune checkpoint not found: {finetune_input}")
    if not donor_input.exists():
        raise FileNotFoundError(f"Donor checkpoint not found: {donor_input}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to overwrite."
        )

    finetune_checkpoint = load_checkpoint(finetune_input)
    donor_checkpoint = load_checkpoint(donor_input)
    patched_checkpoint = patch_checkpoint(finetune_checkpoint, donor_checkpoint)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(patched_checkpoint, output_path)

    print(f"Finetune input : {finetune_input}")
    print(f"Donor input    : {donor_input}")
    print(f"Patched output : {output_path}")


if __name__ == "__main__":
    main()

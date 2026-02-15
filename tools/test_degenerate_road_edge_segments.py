#!/usr/bin/env python3
"""Check degenerate road-edge segments in Nocturne scenario JSON files.

A degenerate segment means two adjacent polyline points are identical
or near-identical within a configurable epsilon.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class SegmentExample:
    """A sample degenerate segment for reporting."""

    road_index: int
    segment_index: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    length: float


@dataclass
class ScenarioReport:
    """Degenerate segment statistics for one scenario."""

    scenario_id: str
    checked_segments: int = 0
    degenerate_segments: int = 0
    examples: List[SegmentExample] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check whether scenario files contain degenerate road-edge segments."
    )
    parser.add_argument(
        "--scenario-data-dir",
        type=str,
        required=True,
        help="Directory containing scenario JSON files.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        action="append",
        default=[],
        help="Specific scenario_id to check. Can be provided multiple times.",
    )
    parser.add_argument(
        "--scenario-index-path",
        type=str,
        default=None,
        help="Optional scenario index JSON with key 'scenario_ids'.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Two adjacent points with distance <= eps are treated as degenerate.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Maximum degenerate examples to keep per scenario.",
    )
    return parser.parse_args()


def _load_index_scenario_ids(index_path: Path) -> List[str]:
    """Load scenario IDs from index file."""
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_ids = data.get("scenario_ids")
    if not isinstance(scenario_ids, list):
        raise ValueError(f"Invalid index format: missing list key 'scenario_ids' in {index_path}")

    return [str(sid) for sid in scenario_ids]


def _normalize_scenario_ids(values: Sequence[str]) -> List[str]:
    """Split comma-separated ids, strip blanks, keep order, and deduplicate."""
    normalized: List[str] = []
    for value in values:
        for token in str(value).split(","):
            scenario_id = token.strip()
            if scenario_id:
                normalized.append(scenario_id)
    return list(dict.fromkeys(normalized))


def _discover_scenario_ids(scenario_data_dir: Path) -> List[str]:
    """Discover all scenario ids by scanning directory."""
    scenario_ids: List[str] = []
    for path in sorted(scenario_data_dir.glob("*.json")):
        name = path.name
        if name.startswith("valid_") or name.endswith("_index.json"):
            continue
        scenario_ids.append(path.stem)
    return scenario_ids


def _resolve_scenario_ids(args: argparse.Namespace, scenario_data_dir: Path) -> List[str]:
    """Resolve final scenario list from args."""
    cli_ids = _normalize_scenario_ids(args.scenario_id)
    if cli_ids:
        return cli_ids

    if args.scenario_index_path:
        index_ids = _load_index_scenario_ids(Path(args.scenario_index_path).expanduser())
        return _normalize_scenario_ids(index_ids)

    return _discover_scenario_ids(scenario_data_dir)


def _parse_point_xy(point: object, context: str) -> Tuple[float, float]:
    """Extract finite x/y from one geometry point."""
    if not isinstance(point, dict):
        raise ValueError(f"{context}: point must be a dict, got {type(point).__name__}")

    if "x" not in point or "y" not in point:
        raise ValueError(f"{context}: point missing x/y keys")

    x = float(point["x"])
    y = float(point["y"])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{context}: point has non-finite coordinate ({x}, {y})")
    return x, y


def inspect_scenario_file(path: Path, scenario_id: str, eps: float, max_examples: int) -> ScenarioReport:
    """Inspect one scenario file and return report."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    roads = data.get("roads")
    if not isinstance(roads, list):
        raise ValueError("invalid scenario format: 'roads' must be a list")

    report = ScenarioReport(scenario_id=scenario_id)
    eps_sq = eps * eps

    for road_index, road in enumerate(roads):
        if not isinstance(road, dict) or road.get("type") != "road_edge":
            continue

        geometry = road.get("geometry")
        if not isinstance(geometry, list) or len(geometry) < 2:
            continue

        for segment_index in range(len(geometry) - 1):
            context = f"{scenario_id}: road[{road_index}] seg[{segment_index}]"
            sx, sy = _parse_point_xy(geometry[segment_index], context)
            ex, ey = _parse_point_xy(geometry[segment_index + 1], context)

            dx = ex - sx
            dy = ey - sy
            len_sq = dx * dx + dy * dy
            length = math.sqrt(len_sq)
            report.checked_segments += 1

            if len_sq <= eps_sq:
                report.degenerate_segments += 1
                if len(report.examples) < max_examples:
                    report.examples.append(
                        SegmentExample(
                            road_index=road_index,
                            segment_index=segment_index,
                            start=(sx, sy),
                            end=(ex, ey),
                            length=length,
                        )
                    )

    return report


def _print_report(
    *,
    scanned_count: int,
    bad_reports: Sequence[ScenarioReport],
    missing_files: Sequence[str],
    parse_errors: Sequence[Tuple[str, str]],
) -> None:
    """Print summary and details."""
    print("=== Degenerate Road-Edge Segment Check ===")
    print(f"Scanned scenarios: {scanned_count}")
    print(f"Scenarios with degenerate segments: {len(bad_reports)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Parse errors: {len(parse_errors)}")

    if bad_reports:
        print("\n--- Scenarios with degenerate segments ---")
        for report in bad_reports:
            print(
                f"{report.scenario_id}: degenerate={report.degenerate_segments}, "
                f"checked_segments={report.checked_segments}"
            )
            for example in report.examples:
                print(
                    "  "
                    f"road[{example.road_index}] seg[{example.segment_index}] "
                    f"start={example.start} end={example.end} len={example.length:.12g}"
                )

    if missing_files:
        print("\n--- Missing scenario files (first 20) ---")
        for scenario_id in missing_files[:20]:
            print(f"  {scenario_id}")

    if parse_errors:
        print("\n--- Parse errors (first 20) ---")
        for scenario_id, err in parse_errors[:20]:
            print(f"  {scenario_id}: {err}")


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument values."""
    if args.eps < 0:
        raise ValueError("--eps must be >= 0")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be >= 0")


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    _validate_args(args)

    scenario_data_dir = Path(args.scenario_data_dir).expanduser()
    if not scenario_data_dir.exists() or not scenario_data_dir.is_dir():
        print(f"Error: invalid --scenario-data-dir: {scenario_data_dir}", file=sys.stderr)
        return 2

    try:
        scenario_ids = _resolve_scenario_ids(args, scenario_data_dir)
    except Exception as exc:
        print(f"Error: failed to resolve scenario ids: {exc}", file=sys.stderr)
        return 2

    if not scenario_ids:
        print("No scenario ids resolved. Nothing to check.")
        return 0

    bad_reports: List[ScenarioReport] = []
    missing_files: List[str] = []
    parse_errors: List[Tuple[str, str]] = []

    for scenario_id in scenario_ids:
        scenario_path = scenario_data_dir / f"{scenario_id}.json"
        if not scenario_path.exists():
            missing_files.append(scenario_id)
            continue

        try:
            report = inspect_scenario_file(
                scenario_path,
                scenario_id=scenario_id,
                eps=args.eps,
                max_examples=args.max_examples,
            )
        except Exception as exc:
            parse_errors.append((scenario_id, str(exc)))
            continue

        if report.degenerate_segments > 0:
            bad_reports.append(report)

    _print_report(
        scanned_count=len(scenario_ids),
        bad_reports=bad_reports,
        missing_files=missing_files,
        parse_errors=parse_errors,
    )

    if missing_files or parse_errors:
        return 2
    if bad_reports:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

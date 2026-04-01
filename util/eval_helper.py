from typing import List, Optional, Sequence


def split_metric_key(metric_key: str) -> tuple[str, str]:
    """Split an eval metric key into metric and environment names."""
    if ":" not in metric_key:
        raise ValueError(
            f"Expected metric key format '<metric_name>:<env_name>', got: {metric_key}"
        )
    metric_name, env_name = metric_key.split(":", 1)
    return metric_name, env_name


def build_eval_csv_headers(accumulator: str | None, value_count: int) -> List[str]:
    """Build semantic CSV headers for eval outputs."""
    base_headers = ["metric_name", "env_name"]
    if accumulator == "mean":
        value_headers = [f"aggregated_mean_{idx}" for idx in range(value_count)]
    else:
        value_headers = [f"episode_{idx}" for idx in range(value_count)]
    return base_headers + value_headers


def build_eval_csv_row(metric_key: str, values: Sequence[float]) -> List[object]:
    """Build one CSV row from a metric key and its recorded values."""
    metric_name, env_name = split_metric_key(metric_key)
    return [metric_name, env_name, *values]


def build_eval_worker_seeds(
    seed: int,
    num_processes: int,
    singleton_env: bool,
) -> List[int]:
    """Build per-worker seeds for evaluation environments."""
    base_seed = int(seed)
    if singleton_env:
        return [base_seed] * num_processes
    return [base_seed + idx for idx in range(num_processes)]


def set_eval_worker_seeds(
    venv,
    seed: Optional[int],
    num_processes: int,
    singleton_env: bool,
):
    """Apply deterministic per-worker seeds to a vectorized eval environment."""
    if seed is None:
        return None
    seeds = build_eval_worker_seeds(
        seed=seed,
        num_processes=num_processes,
        singleton_env=singleton_env,
    )
    return venv.set_seed(seeds)

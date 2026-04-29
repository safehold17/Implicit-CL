"""ClearML dataset and artifact helpers."""

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def download_clearml_dataset(dataset_project: str, dataset_name: str) -> str:
    """Download a ClearML dataset and return the local path."""
    from clearml import Dataset as ClearMLDataset

    try:
        local_copy = ClearMLDataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
        ).get_local_copy()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download ClearML dataset "
            f"(project={dataset_project!r}, name={dataset_name!r})"
        ) from e
    log.info("Downloaded ClearML dataset to: %s", local_copy)
    return local_copy


def upload_clearml_artifact(
    clearml_task: Any,
    artifact_path: str,
    artifact_name: str | None = None,
) -> None:
    """Upload one local file to ClearML artifacts."""
    if clearml_task is None:
        return

    resolved_name = artifact_name or os.path.basename(artifact_path)
    try:
        clearml_task.upload_artifact(
            name=resolved_name,
            artifact_object=artifact_path,
            wait_on_upload=True,
        )
    except Exception:
        log.warning(
            "Failed to upload ClearML artifact: %s",
            artifact_path,
            exc_info=True,
        )


def get_clearml_logger(clearml_task: Any = None) -> Any:
    """Return the ClearML logger only when an explicit task is available."""
    if clearml_task is not None:
        return clearml_task.get_logger()
    return None


def report_clearml_scalar(
    clearml_logger: Any,
    tag: str,
    value: float,
    iteration: int,
) -> None:
    """Report one scalar to ClearML using ``section/name`` tag semantics."""
    if clearml_logger is None:
        return

    if "/" in tag:
        title, series = tag.split("/", 1)
    else:
        title, series = "metrics", tag

    try:
        clearml_logger.report_scalar(
            title=title,
            series=series,
            value=float(value),
            iteration=int(iteration),
        )
    except Exception:
        log.warning(
            "Failed to report ClearML scalar: %s=%s at step %s",
            tag,
            value,
            iteration,
            exc_info=True,
        )


def finalize_clearml_run(
    filewriter: Any,
    clearml_task: Any,
    successful: bool,
) -> None:
    """Close the writer and upload the final ``meta.json`` when ClearML is active."""
    try:
        filewriter.close(successful=successful)
    except Exception:
        log.warning("Failed to close FileWriter before ClearML finalization", exc_info=True)
        return

    upload_clearml_artifact(clearml_task, filewriter.paths["meta"])

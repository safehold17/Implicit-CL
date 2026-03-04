"""ClearML dataset helpers."""

import logging

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

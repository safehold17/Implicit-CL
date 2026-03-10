"""
IPC payload 编解码（batch_inference 专用）。
IPC payload encoding and decoding for batch_inference.
"""

from __future__ import annotations

from .ipc.model_outputs import pack_model_outputs, unpack_model_outputs
from .ipc.prepared import pack_prepared, release_prepared_payload, unpack_prepared
from .ipc.schema import MODEL_OUTPUTS_IPC_FORMAT, PREPARED_IPC_FORMAT
from .ipc.validate import validate_model_outputs_payload, validate_prepared_payload

__all__ = [
    "MODEL_OUTPUTS_IPC_FORMAT",
    "PREPARED_IPC_FORMAT",
    "pack_model_outputs",
    "pack_prepared",
    "release_prepared_payload",
    "unpack_model_outputs",
    "unpack_prepared",
    "validate_model_outputs_payload",
    "validate_prepared_payload",
]

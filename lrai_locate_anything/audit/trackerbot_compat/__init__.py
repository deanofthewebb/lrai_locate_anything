"""trackerbot_compat — vendored canonical SafeTunnel ByteTracker + adapter.

Vendored from /mnt/ssd1/tracker_bot/src/safetunnel/core/tracking/bytetrack.py
at commit referenced in SOURCE.md.  The tracker is imported by lrai_isp_audit.py
when --tracker trackerbot is passed.  No safetunnel package dependency — all
required types are re-implemented here.
"""
from .bytetrack import (
    ByteTrackerConfig,
    ByteTrackerAuditWrapper,
    STrack,
    KalmanFilter,
    _iou_batch,
    _linear_assignment,
)

__all__ = [
    "ByteTrackerConfig",
    "ByteTrackerAuditWrapper",
    "STrack",
    "KalmanFilter",
    "_iou_batch",
    "_linear_assignment",
]

# trackerbot_compat — vendoring source

## Origin

Vendored from the canonical SafeTunnel tracker at:

    /mnt/ssd1/tracker_bot/src/safetunnel/core/tracking/bytetrack.py

Local mirror used during vendoring (same code, same content):

    /Users/deanwebb/Development/tracker_bot/src/safetunnel/core/tracking/bytetrack.py

Classes copied verbatim (all in `bytetrack.py`):

| Class / function      | Source lines (approx) | Notes |
|-----------------------|-----------------------|-------|
| `ByteTrackerConfig`   | 28–39                 | Dataclass, verbatim |
| `ByteTrack`           | 12–24                 | Output dataclass, verbatim |
| `STrack`              | 42–145                | Per-track Kalman wrapper, verbatim |
| `KalmanFilter`        | 147–214               | 8-dim XYAH, verbatim |
| `_iou_batch`          | 217–238               | Verbatim |
| `_linear_assignment`  | 241–265               | Verbatim (lap + scipy fallback) |
| `ByteTracker.update`  | 290–474               | Verbatim logic; `measurements` /
|                       |                       | `ProjectedMeasurement` path removed |

## Changes vs source

1. `safetunnel.core.types.{BBox,Detection,ProjectedMeasurement}` replaced by
   local `_BBox` / `_Detection` stubs so there is no runtime dependency on the
   safetunnel package inside the audit container.

2. `ByteTracker._update_depth_from_measurements` removed (needs
   `ProjectedMeasurement`; unused in 2D audit context).

3. Public class renamed `_CanonicalByteTracker` (internal); the external
   surface is `ByteTrackerAuditWrapper` which adapts the audit per-frame loop's
   `List[((x1,y1,x2,y2), label_str)]` input to canonical `List[_Detection]`
   and converts the `List[ByteTrack]` output to
   `List[(track_id, label, prev_center, curr_center)]`.

## Rationale

The audit script needs a tracker that matches tracker_bot prod quality
(2-stage association, Kalman identity preservation) without importing the
full safetunnel package into the TRT-LLM audit container.  Vendoring the
numpy-only core avoids the Rust/Docker dependency chain while keeping the
counting-line crossing logic bitwise-identical to prod.

## Dependency check

- `numpy` — hard dep of audit script, always present.
- `lap` (optional) — preferred LAP solver; graceful fallback to
  `scipy.optimize.linear_sum_assignment` (supervision transitive dep).
- No `ultralytics`, no `torch`, no `safetunnel` package required.

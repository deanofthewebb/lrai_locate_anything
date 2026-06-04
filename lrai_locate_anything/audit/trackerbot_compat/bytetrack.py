"""Vendored canonical SafeTunnel ByteTracker — numpy-only, no safetunnel dep.

Source: /mnt/ssd1/tracker_bot/src/safetunnel/core/tracking/bytetrack.py
        (class ByteTracker, ByteTrackerConfig, STrack, KalmanFilter,
         _iou_batch, _linear_assignment)
See SOURCE.md for commit reference and vendoring rationale.

The only public addition vs the source is ByteTrackerAuditWrapper — a thin
adapter that converts the audit per-frame loop's
    List[((x1,y1,x2,y2), class_label_str)]
inputs into the canonical ByteTracker.update() signature and re-shapes the
returned List[ByteTrack] objects back into the
    List[(track_id:int, class_label:str, prev_center:(x,y), curr_center:(x,y))]
tuples that _evaluate_line expects.

scipy.optimize.linear_sum_assignment is used as the LAP fallback (lap is
preferred if installed, matching canonical behaviour).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Minimal BBox + Detection stubs (replace safetunnel.core.types dependency)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _BBox:
    x: int
    y: int
    w: int
    h: int

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h))

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> "_BBox":
        return _BBox(x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1))


@dataclass(frozen=True)
class _Detection:
    bbox: _BBox
    score: float
    class_id: int
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# ByteTrackerConfig (canonical dataclass, verbatim)
# ---------------------------------------------------------------------------

@dataclass
class ByteTrackerConfig:
    # Mirrors ultralytics BYTETracker args.  track_thresh is kept as a
    # back-compat alias for track_high_thresh.
    track_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    min_box_area: int = 10
    frame_rate: int = 30


# ---------------------------------------------------------------------------
# ByteTrack output dataclass (canonical, verbatim — minus safetunnel imports)
# ---------------------------------------------------------------------------

@dataclass
class ByteTrack:
    track_id: int
    s: float
    v: float
    last_ts: float
    age: int = 0
    missed: int = 0
    confidence: float = 0.0
    bbox_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)
    lane_u: float = float("nan")
    anchor_xy: Tuple[int, int] = (0, 0)
    tlwh: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    score: float = 0.0


# ---------------------------------------------------------------------------
# STrack (canonical, verbatim)
# ---------------------------------------------------------------------------

class STrack:
    shared_kalman = None
    _count = 0

    def __init__(self, tlwh: np.ndarray, score: float):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.state = 0
        self.idx = 0
        self.frame_id = 0
        self.start_frame = 0
        self.track_id = 0
        self.s = 0.0
        self.v = 0.0
        self.lane_u: float = float("nan")
        self.anchor_xy: Tuple[int, int] = (0, 0)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        tlbr = self.tlbr
        if any(math.isnan(v) or math.isinf(v) for v in tlbr):
            return (0, 0, 0, 0)
        return (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))

    def activate(self, kalman_filter, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        STrack._count += 1
        self.track_id = STrack._count
        self.mean, self.covariance = self.kalman_filter.initiate(self._tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = 1
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = 1
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            STrack._count += 1
            self.track_id = STrack._count
        self.score = new_track.score

    def update(self, new_track: "STrack", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_tlwh)
        )
        self.state = 1
        self.is_activated = True
        self.score = new_track.score

    def predict(self) -> None:
        if self.mean is not None and self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    @staticmethod
    def _tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def mark_lost(self) -> None:
        self.state = 2

    def mark_removed(self) -> None:
        self.state = 3


# ---------------------------------------------------------------------------
# KalmanFilter (canonical XYAH 8-dim, verbatim)
# ---------------------------------------------------------------------------

class KalmanFilter:
    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        chol_factor = np.linalg.cholesky(projected_cov)
        cross = np.dot(covariance, self._update_mat.T)
        kalman_gain = np.linalg.solve(
            chol_factor.T, np.linalg.solve(chol_factor, cross.T)
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


# ---------------------------------------------------------------------------
# Private helpers (canonical, verbatim)
# ---------------------------------------------------------------------------

def _iou_batch(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    for i, a in enumerate(atlbrs):
        for j, b in enumerate(btlbrs):
            xx1 = max(a[0], b[0])
            yy1 = max(a[1], b[1])
            xx2 = min(a[2], b[2])
            yy2 = min(a[3], b[3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            ious[i, j] = inter / union if union > 0 else 0.0
    return ious


def _linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List, List, List]:
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = [ix for ix, mx in enumerate(x) if mx < 0]
        unmatched_b = [ix for ix, mx in enumerate(y) if mx < 0]
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c])
                if r in unmatched_a:
                    unmatched_a.remove(r)
                if c in unmatched_b:
                    unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# Canonical ByteTracker (verbatim from source, measurements path removed
# since ProjectedMeasurement is a safetunnel-only concept unused in audit)
# ---------------------------------------------------------------------------

class _CanonicalByteTracker:
    """Internal — use ByteTrackerAuditWrapper externally."""

    def __init__(self, cfg: ByteTrackerConfig = ByteTrackerConfig()):
        self.cfg = cfg
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        self.max_time_lost = int(cfg.frame_rate / 30.0 * cfg.track_buffer)
        self.kalman_filter = KalmanFilter()
        self._depth_map: dict = {}
        STrack._count = 0

    def update(self, detections: List[_Detection], ts: float = 0.0) -> List[ByteTrack]:
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        if not detections:
            for track in self.tracked_stracks:
                track.mark_lost()
                lost_stracks.append(track)
            self.tracked_stracks = []
            self.lost_stracks = list(set(self.lost_stracks + lost_stracks))
            still_lost = []
            for t in self.lost_stracks:
                if self.frame_id - t.frame_id > self.max_time_lost:
                    t.mark_removed()
                    removed_stracks.append(t)
                else:
                    still_lost.append(t)
            self.lost_stracks = still_lost
            self.removed_stracks.extend(removed_stracks)
            if len(self.removed_stracks) > 1000:
                self.removed_stracks = self.removed_stracks[-1000:]
            return self._to_output(ts)

        det_bboxes = []
        det_scores = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox.to_xyxy()
            det_bboxes.append([x1, y1, x2 - x1, y2 - y1])
            det_scores.append(d.score)

        det_bboxes = np.array(det_bboxes, dtype=np.float32)
        det_scores = np.array(det_scores, dtype=np.float32)

        remain_inds = det_scores >= self.cfg.track_thresh
        inds_low = det_scores > self.cfg.track_low_thresh
        inds_high = det_scores < self.cfg.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = det_bboxes[inds_second]
        scores_second = det_scores[inds_second]
        dets = det_bboxes[remain_inds]
        scores_keep = det_scores[remain_inds]

        detections_high = [STrack(tlwh, s) for tlwh, s in zip(dets, scores_keep)]
        detections_low = [STrack(tlwh, s) for tlwh, s in zip(dets_second, scores_second)]

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]
        strack_pool = tracked_stracks + self.lost_stracks
        for t in strack_pool:
            t.predict()

        if strack_pool and detections_high:
            track_tlbrs = np.array([t.tlbr for t in strack_pool])
            det_tlbrs = np.array([d.tlbr for d in detections_high])
            ious = _iou_batch(track_tlbrs, det_tlbrs)
            dists = 1 - ious
            if self.cfg.fuse_score and dists.size > 0:
                iou_sim = 1 - dists
                det_scores_keep = np.array([d.score for d in detections_high])
                det_scores_keep = det_scores_keep[None].repeat(dists.shape[0], axis=0)
                dists = 1 - (iou_sim * det_scores_keep)
            matches, u_track, u_detection = _linear_assignment(dists, thresh=self.cfg.match_thresh)
        else:
            matches, u_track, u_detection = [], list(range(len(strack_pool))), list(range(len(detections_high)))

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if track.state != 2:
                track.mark_lost()
                lost_stracks.append(track)

        detections_second_remaining = list(detections_low)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 1]

        if r_tracked_stracks and detections_second_remaining:
            track_tlbrs = np.array([t.tlbr for t in r_tracked_stracks])
            det_tlbrs = np.array([d.tlbr for d in detections_second_remaining])
            ious = _iou_batch(track_tlbrs, det_tlbrs)
            dists = 1 - ious
            matches, u_track_2, _ = _linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second_remaining[idet]
                if track.state == 1:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            for it in u_track_2:
                track = r_tracked_stracks[it]
                if track.state != 2:
                    track.mark_lost()
                    lost_stracks.append(track)

        dets_unc_remaining = [detections_high[i] for i in u_detection]
        idx_map_unc = list(u_detection)
        if unconfirmed and dets_unc_remaining:
            track_tlbrs = np.array([t.tlbr for t in unconfirmed])
            det_tlbrs = np.array([d.tlbr for d in dets_unc_remaining])
            ious = _iou_batch(track_tlbrs, det_tlbrs)
            dists = 1 - ious
            if self.cfg.fuse_score and dists.size > 0:
                iou_sim = 1 - dists
                det_scores_unc = np.array([d.score for d in dets_unc_remaining])
                det_scores_unc = det_scores_unc[None].repeat(dists.shape[0], axis=0)
                dists = 1 - (iou_sim * det_scores_unc)
            matches_unc, u_unconfirmed, u_det_unc = _linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches_unc:
                unconfirmed[itracked].update(dets_unc_remaining[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
            u_detection = [idx_map_unc[i] for i in u_det_unc]
        else:
            for t in unconfirmed:
                t.mark_removed()
                removed_stracks.append(t)

        for inew in u_detection:
            track = detections_high[inew]
            if track.score < self.cfg.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.tracked_stracks = list(set(self.tracked_stracks + activated_stracks + refind_stracks))
        self.lost_stracks = list(set([t for t in self.lost_stracks if t.state == 2] + lost_stracks))
        removed_ids = {t.track_id for t in removed_stracks}
        self.lost_stracks = [t for t in self.lost_stracks if t.track_id not in removed_ids]
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return self._to_output(ts)

    def _to_output(self, ts: float) -> List[ByteTrack]:
        output = []
        for track in self.tracked_stracks:
            if track.is_activated:
                output.append(
                    ByteTrack(
                        track_id=track.track_id,
                        s=track.s,
                        v=track.v,
                        last_ts=ts,
                        age=track.tracklet_len,
                        missed=0,
                        confidence=track.score,
                        bbox_xyxy=track.xyxy,
                        lane_u=track.lane_u,
                        anchor_xy=track.anchor_xy,
                        tlwh=tuple(track.tlwh.tolist()),
                        score=track.score,
                    )
                )
        return sorted(output, key=lambda t: t.s)


# ---------------------------------------------------------------------------
# Audit adapter wrapper — public surface for lrai_isp_audit.py
# ---------------------------------------------------------------------------

class ByteTrackerAuditWrapper:
    """Canonical SafeTunnel ByteTracker wrapped to match the audit per-frame
    loop's call contract:

        update(detections: List[((x1,y1,x2,y2), class_label_str)], frame_idx: int)
          -> List[(track_id, class_label, prev_center, curr_center)]

    Detections are given score=1.0 because the VLM detector does not produce
    confidence values — every detection clears the high-confidence threshold.

    Class labels are tracked per track_id using latest-wins semantics (same
    as the existing ByteTracker wrapper in the audit script).
    """

    def __init__(self, frame_rate: int = 5):
        cfg = ByteTrackerConfig(
            frame_rate=frame_rate,
            track_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
        self._tracker = _CanonicalByteTracker(cfg)
        self._prev_centers: Dict[int, Tuple[float, float]] = {}
        self._class_by_tid: Dict[int, str] = {}
        self._label_to_id: Dict[str, int] = {}

    def _intern(self, label: str) -> int:
        cid = self._label_to_id.get(label)
        if cid is None:
            cid = len(self._label_to_id)
            self._label_to_id[label] = cid
        return cid

    def update(
        self,
        detections: List[Tuple[Tuple[float, float, float, float], str]],
        frame_idx: int,
    ) -> List[Tuple[int, str, Tuple[float, float], Tuple[float, float]]]:
        if not detections:
            return []

        det_objs: List[_Detection] = []
        for (x1, y1, x2, y2), label in detections:
            bbox = _BBox.from_xyxy(x1, y1, x2, y2)
            cid = self._intern(label)
            det_objs.append(_Detection(bbox=bbox, score=1.0, class_id=cid, label=label))

        ts = float(frame_idx)
        byte_tracks: List[ByteTrack] = self._tracker.update(det_objs, ts=ts)

        results: List[Tuple[int, str, Tuple[float, float], Tuple[float, float]]] = []
        for bt in byte_tracks:
            tid = bt.track_id
            x1, y1, x2, y2 = bt.bbox_xyxy
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            # Recover label: prefer the label from the detection that last
            # updated this track (stored in _class_by_tid); fall back to
            # "unknown" if no detection-label mapping is available yet.
            label = self._class_by_tid.get(tid, "unknown")
            prev = self._prev_centers.get(tid, (cx, cy))
            self._prev_centers[tid] = (cx, cy)
            results.append((tid, label, prev, (cx, cy)))

        # Update label map from current frame's detections.  We do this after
        # the tracker update so the label reflects the frame that just matched,
        # not the frame before.  Since ByteTracker does not expose which input
        # detection matched which output track (by design), we use a proximity
        # heuristic: for each active track, find the closest detection center.
        if detections:
            det_centers = [
                ((x1 + x2) / 2.0, (y1 + y2) / 2.0, lbl)
                for (x1, y1, x2, y2), lbl in detections
            ]
            for bt in byte_tracks:
                tid = bt.track_id
                x1, y1, x2, y2 = bt.bbox_xyxy
                tcx, tcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                best_lbl = "unknown"
                best_d2 = float("inf")
                for dcx, dcy, dlbl in det_centers:
                    d2 = (tcx - dcx) ** 2 + (tcy - dcy) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_lbl = dlbl
                self._class_by_tid[tid] = best_lbl

        return results

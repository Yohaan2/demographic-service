import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from src.core.config import settings
from src.core.logging import logger


class STrack:
    """Single Track representation using a Constant-Velocity Kalman Filter."""
    _count = 0
    _lock = threading.Lock()

    def __init__(self, bbox: np.ndarray, score: float):
        # Thread-safe track ID increment
        with STrack._lock:
            STrack._count += 1
            self.track_id = STrack._count

        self.score = score
        self.state = 1  # 1: Tracked, 2: Lost, 3: Removed
        self.is_activated = False
        
        # Kalman filter state initialization: x = [cx, cy, s, r, vx, vy, vs]
        # cx, cy: center coordinates, s: size (area), r: aspect ratio (w/h)
        # vx, vy, vs: velocities
        self.mean, self.covariance = self._init_kalman(bbox)
        
        # Keep track of history and longevity
        self.tracklet_len = 0
        self.start_frame = 0
        self.last_seen_timestamp = time.time()

    def _init_kalman(self, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes the Kalman Filter mean and covariance matrices for the bbox."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = float(w) / h if h > 0 else 1.0

        # State vector: [cx, cy, s, r, vx, vy, vs]
        mean = np.zeros(7, dtype=np.float32)
        mean[:4] = [cx, cy, s, r]

        # Covariance matrices
        # High uncertainty in initial velocities
        std_pos = [
            2 * 0.05 * h,
            2 * 0.05 * h,
            1e-2,
            0.05
        ]
        std_vel = [
            10 * 0.05 * h,
            10 * 0.05 * h,
            1e-5
        ]
        
        # Initial state covariance matrix (P)
        covariance = np.diag(np.square(np.hstack([std_pos, std_vel]))).astype(np.float32)
        return mean, covariance

    @property
    def tlbr(self) -> np.ndarray:
        """Returns bounding box in Top-Left Bottom-Right [x1, y1, x2, y2] format."""
        cx, cy, s, r = self.mean[:4]
        
        # Handle division by zero or negative values
        if s <= 0 or r <= 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)
            
        h = np.sqrt(s / r)
        w = r * h
        
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self) -> None:
        """Predicts the state in the next frame (Kalman State Transition step)."""
        # Constant velocity transition matrices
        dt = 1.0
        F = np.eye(7, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt

        # Process noise covariance (Q)
        h = np.sqrt(self.mean[2] / self.mean[3]) if self.mean[2] > 0 and self.mean[3] > 0 else 100.0
        std_pos = [
            0.05 * h,
            0.05 * h,
            1e-2,
            0.05
        ]
        std_vel = [
            0.01 * h,
            0.01 * h,
            1e-5
        ]
        Q = np.diag(np.square(np.hstack([std_pos, std_vel]))).astype(np.float32)

        # x' = F * x
        self.mean = np.dot(F, self.mean)
        # P' = F * P * F^T + Q
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def update(self, bbox: np.ndarray, score: float) -> None:
        """Updates the track state with a new detection (Kalman Update/Correction step)."""
        self.score = score
        self.tracklet_len += 1
        self.last_seen_timestamp = time.time()
        self.state = 1  # Active / Tracked
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = float(w) / h if h > 0 else 1.0
        
        # Measurement vector: z = [cx, cy, s, r]
        z = np.array([cx, cy, s, r], dtype=np.float32)

        # Measurement matrix (H) - maps state [7] to measurement [4]
        H = np.zeros((4, 7), dtype=np.float32)
        H[:4, :4] = np.eye(4, dtype=np.float32)

        # Measurement noise covariance (R)
        std_meas = [
            0.05 * h,
            0.05 * h,
            1e-1,
            0.05
        ]
        R = np.diag(np.square(std_meas)).astype(np.float32)

        # Kalman gain calculation: K = P * H^T * (H * P * H^T + R)^-1
        P_HT = np.dot(self.covariance, H.T)
        S = np.dot(H, P_HT) + R
        K = np.dot(P_HT, np.linalg.inv(S))

        # Update mean and covariance
        self.mean = self.mean + np.dot(K, (z - np.dot(H, self.mean)))
        self.covariance = self.covariance - np.dot(K, np.dot(H, self.covariance))


class ByteTracker:
    """ByteTrack algorithm tracking instance."""
    def __init__(self, iou_threshold: float = 0.01):  # Reducido al 1% para cámaras de bajos FPS o saltos espaciales amplios
        self.tracked_tracks: List[STrack] = []  # Active tracks
        self.lost_tracks: List[STrack] = []     # Temporarily lost tracks
        self.iou_threshold = iou_threshold
        self.max_lost_frames = 30  # Keep lost tracks for up to 30 frames
        self.frame_count = 0

    @staticmethod
    def _inflate_boxes(boxes: np.ndarray, factor: float = 0.8) -> np.ndarray:
        """Inflates bounding boxes to handle wide spatial jumps in low FPS cameras."""
        if boxes.size == 0:
            return boxes
        inflated = boxes.copy()
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        inflated[:, 0] -= w * factor
        inflated[:, 1] -= h * factor
        inflated[:, 2] += w * factor
        inflated[:, 3] += h * factor
        return inflated

    @staticmethod
    def _compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Computes pairwise Intersection-over-Union (IoU) matrix between A and B."""
        if boxes_a.size == 0 or boxes_b.size == 0:
            return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

        # A [N, 4], B [M, 4] -> Output [N, M]
        N, M = boxes_a.shape[0], boxes_b.shape[0]
        iou_matrix = np.zeros((N, M), dtype=np.float32)

        for i in range(N):
            box_a = boxes_a[i]
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            for j in range(M):
                box_b = boxes_b[j]
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

                # Intersection bounds
                xi1 = max(box_a[0], box_b[0])
                yi1 = max(box_a[1], box_b[1])
                xi2 = min(box_a[2], box_b[2])
                yi2 = min(box_a[3], box_b[3])

                inter_w = max(0.0, xi2 - xi1)
                inter_h = max(0.0, yi2 - yi1)
                inter_area = inter_w * inter_h

                union_area = area_a + area_b - inter_area
                if union_area > 0:
                    iou_matrix[i, j] = inter_area / union_area

        return iou_matrix

    def update(self, bboxes: np.ndarray, scores: np.ndarray) -> List[STrack]:
        """
        Updates the tracker with new detections in the current frame.
        
        Args:
            bboxes: Bboxes array [N, 4] as [x1, y1, x2, y2].
            scores: Confidences array [N].
            
        Returns:
            A list of currently tracked active STrack objects.
        """
        self.frame_count += 1
        
        # 1. Split detections into high-score and low-score pools
        high_idx = np.where(scores >= 0.6)[0]
        low_idx = np.where((scores >= 0.1) & (scores < 0.6))[0]
        
        det_high = bboxes[high_idx] if len(high_idx) > 0 else np.empty((0, 4))
        scores_high = scores[high_idx] if len(high_idx) > 0 else np.empty(0)
        
        det_low = bboxes[low_idx] if len(low_idx) > 0 else np.empty((0, 4))
        scores_low = scores[low_idx] if len(low_idx) > 0 else np.empty(0)
        
        # 2. Kalman Filter state prediction step for all active and lost tracks
        all_tracks = self.tracked_tracks + self.lost_tracks
        for track in all_tracks:
            track.predict()
            
        # 3. FIRST ASSOCIATION: High-score detections with active tracks
        # Prepare predicted track boxes and inflate them for tolerant spatial search
        predicted_boxes = np.array([t.tlbr for t in self.tracked_tracks]) if len(self.tracked_tracks) > 0 else np.empty((0, 4))
        inflated_predicted = self._inflate_boxes(predicted_boxes, factor=0.8)
        iou_matrix = self._compute_iou(inflated_predicted, det_high)
        
        # Apply Hungarian algorithm (minimizing cost = 1.0 - IoU)
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks_1 = []
        unmatched_tracks = set(range(len(self.tracked_tracks)))
        unmatched_dets_high = set(range(len(det_high)))
        
        for r, c in zip(row_ind, col_ind):
            # Enforce strict IoU threshold
            if iou_matrix[r, c] >= self.iou_threshold:
                track = self.tracked_tracks[r]
                det_idx = high_idx[c]
                track.update(bboxes[det_idx], scores[det_idx])
                matched_tracks_1.append(track)
                unmatched_tracks.discard(r)
                unmatched_dets_high.discard(c)

        # 4. SECOND ASSOCIATION: Low-score detections with remaining active tracks
        remaining_tracked = [self.tracked_tracks[i] for i in unmatched_tracks]
        predicted_remaining_boxes = np.array([t.tlbr for t in remaining_tracked]) if len(remaining_tracked) > 0 else np.empty((0, 4))
        inflated_remaining = self._inflate_boxes(predicted_remaining_boxes, factor=0.8)
        
        iou_matrix_low = self._compute_iou(inflated_remaining, det_low)
        cost_matrix_low = 1.0 - iou_matrix_low
        row_ind_low, col_ind_low = linear_sum_assignment(cost_matrix_low)
        
        matched_tracks_2 = []
        unmatched_tracked_after_2 = set(range(len(remaining_tracked)))
        
        for r, c in zip(row_ind_low, col_ind_low):
            if iou_matrix_low[r, c] >= self.iou_threshold:  # Utilizar el umbral tolerante dinámico
                track = remaining_tracked[r]
                det_idx = low_idx[c]
                track.update(bboxes[det_idx], scores[det_idx])
                matched_tracks_2.append(track)
                unmatched_tracked_after_2.discard(r)

        # 5. THIRD ASSOCIATION: Unmatched remaining active tracks with lost tracks (for re-activation)
        lost_predicted_boxes = np.array([t.tlbr for t in self.lost_tracks]) if len(self.lost_tracks) > 0 else np.empty((0, 4))
        inflated_lost = self._inflate_boxes(lost_predicted_boxes, factor=0.8)
        unmatched_dets_high_list = list(unmatched_dets_high)
        det_high_remaining = det_high[unmatched_dets_high_list] if len(unmatched_dets_high) > 0 else np.empty((0, 4))
        
        iou_matrix_lost = self._compute_iou(inflated_lost, det_high_remaining)
        cost_matrix_lost = 1.0 - iou_matrix_lost
        row_ind_lost, col_ind_lost = linear_sum_assignment(cost_matrix_lost)
        
        matched_lost_tracks = []
        unmatched_lost = set(range(len(self.lost_tracks)))
        
        for r, c in zip(row_ind_lost, col_ind_lost):
            if iou_matrix_lost[r, c] >= self.iou_threshold:
                track = self.lost_tracks[r]
                det_idx = high_idx[unmatched_dets_high_list[c]]
                track.update(bboxes[det_idx], scores[det_idx])
                matched_lost_tracks.append(track)
                unmatched_lost.discard(r)
                unmatched_dets_high.discard(unmatched_dets_high_list[c])

        # 6. INITIALIZE NEW TRACKS: Remaining high-score detections
        new_tracks = []
        for c in unmatched_dets_high:
            det_idx = high_idx[c]
            new_track = STrack(bboxes[det_idx], scores[det_idx])
            new_track.is_activated = True
            new_tracks.append(new_track)
            
        # 7. MANAGE STATES & LIFECYCLE
        # Lost tracks transition
        new_lost_tracks = []
        for idx in unmatched_tracked_after_2:
            track = remaining_tracked[idx]
            track.state = 2  # Lost
            new_lost_tracks.append(track)
            
        # Clean lost tracks exceeding max lost threshold (30 updates)
        still_lost_tracks = []
        for idx in unmatched_lost:
            track = self.lost_tracks[idx]
            # If a track is lost for too long, remove it
            if track.tracklet_len < self.max_lost_frames:
                track.tracklet_len += 1
                still_lost_tracks.append(track)
                
        # Re-assemble active and lost tracks lists
        self.tracked_tracks = matched_tracks_1 + matched_tracks_2 + matched_lost_tracks + new_tracks
        self.lost_tracks = new_lost_tracks + still_lost_tracks
        
        return self.tracked_tracks


class TrackerRegistry:
    """Thread-safe Multi-camera Tracker Singleton Registry."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(TrackerRegistry, cls).__new__(cls, *args, **kwargs)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._trackers: Dict[str, Tuple[ByteTracker, float]] = {}  # camera_id -> (tracker, last_access_time)
        self._lock = threading.Lock()
        self._initialized = True
        
        # Periodic cleanup daemon thread to prevent memory leak
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def get_tracker(self, camera_id: str) -> ByteTracker:
        """Retrieves an existing tracker instance for the camera or creates a new one on demand."""
        with self._lock:
            now = time.time()
            if camera_id in self._trackers:
                tracker, _ = self._trackers[camera_id]
                self._trackers[camera_id] = (tracker, now)
                return tracker
                
            # Create on demand if under the maximum capacity limit
            if len(self._trackers) >= settings.MAX_TRACKERS:
                self._force_cleanup_locked()
                
            tracker = ByteTracker()
            self._trackers[camera_id] = (tracker, now)
            logger.info("Created new ByteTracker instance on registry", camera_id=camera_id)
            return tracker

    def get_active_trackers_count(self) -> int:
        """Returns the current number of allocated trackers."""
        with self._lock:
            return len(self._trackers)

    def _force_cleanup_locked(self) -> None:
        """Prunes the single oldest unused tracker when capacity is exceeded."""
        if not self._trackers:
            return
        # Find tracker with the oldest last access timestamp
        oldest_cam = min(self._trackers.keys(), key=lambda k: self._trackers[k][1])
        del self._trackers[oldest_cam]
        logger.warning("Max trackers reached. Evicted oldest inactive tracker from registry", camera_id=oldest_cam)

    def _cleanup_loop(self) -> None:
        """Background daemon execution loop to clean expired trackers periodically."""
        while True:
            time.sleep(30)  # Check every 30 seconds
            try:
                self.cleanup_expired()
            except Exception as e:
                logger.error("Error during tracker registry auto-cleanup", error=str(e))

    def cleanup_expired(self) -> None:
        """Removes cameras whose trackers have been inactive for longer than TTL."""
        with self._lock:
            now = time.time()
            expired_cams = []
            for camera_id, (_, last_seen) in self._trackers.items():
                if now - last_seen > settings.TRACKER_TTL:
                    expired_cams.append(camera_id)
                    
            for camera_id in expired_cams:
                del self._trackers[camera_id]
                logger.info(
                    "Evicted inactive camera tracker from registry due to TTL expiration", 
                    camera_id=camera_id, 
                    ttl=settings.TRACKER_TTL
                )


# Instancia única Singleton global
tracker_registry = TrackerRegistry()

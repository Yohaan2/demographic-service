from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from src.trackers.bytetrack_manager import tracker_registry

# 1. Total Requests Counter (labeled by camera ID and HTTP Status)
REQUESTS_TOTAL = Counter(
    "demographics_requests_total",
    "Total number of HTTP analysis requests received",
    ["camera_id", "status"]
)

# 2. Frame processing time Histogram
PROCESSING_TIME = Histogram(
    "demographics_processing_time_seconds",
    "Time spent executing the full deep learning pipeline",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
)

# 3. Faces Detected per frame Histogram
FACES_DETECTED = Histogram(
    "demographics_faces_detected_count",
    "Distribution of face detections per processed frame",
    buckets=(0, 1, 2, 3, 5, 10, 20)
)

# 4. Tracker count Gauge (Dynamic mapping via registry)
TRACKER_COUNT = Gauge(
    "demographics_tracker_count",
    "Current active tracking instances in memory registry"
)

# 5. Active cameras Gauge
ACTIVE_CAMERAS = Gauge(
    "demographics_active_cameras",
    "Current active cameras with tracking buffers"
)


def get_prometheus_metrics() -> tuple[bytes, str]:
    """
    Updates dynamic gauges from memory registries and generates 
    latest prometheus exposition payload.
    """
    # Dynamically query current registry capacity before rendering metrics
    active_count = tracker_registry.get_active_trackers_count()
    TRACKER_COUNT.set(active_count)
    ACTIVE_CAMERAS.set(active_count)
    
    return generate_latest(), CONTENT_TYPE_LATEST

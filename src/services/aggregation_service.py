import time
import threading
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, List
from src.core.config import settings
from src.core.logging import logger


class TemporalAggregationService:
    """Thread-safe sliding-window temporal aggregation service for age and gender tracks."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(TemporalAggregationService, cls).__new__(cls, *args, **kwargs)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._history: Dict[int, Dict] = {}  # track_id -> {"window": deque, "last_updated": float}
        self._lock = threading.Lock()
        self._initialized = True
        
        # Start periodic background garbage collector
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    @staticmethod
    def get_age_range(age: float) -> str:
        """Categorizes continuous age values into standard sociological marketing ranges."""
        age_int = int(round(age))
        if age_int <= 2:
            return "0-2"
        elif age_int <= 12:
            return "3-12"
        elif age_int <= 19:
            return "13-19"
        elif age_int <= 24:
            return "20-24"
        elif age_int <= 34:
            return "25-34"
        elif age_int <= 44:
            return "35-44"
        elif age_int <= 54:
            return "45-54"
        elif age_int <= 64:
            return "55-64"
        else:
            return "65+"

    def update_and_aggregate(
        self, 
        track_id: int, 
        age: float, 
        gender: str, 
        gender_confidence: float
    ) -> Tuple[int, str, float, str]:
        """
        Pushes a new sample to the sliding window of track_id and returns the consolidated metrics.
        
        Args:
            track_id: ID of the face track.
            age: Estimated age for current frame.
            gender: Estimated gender ('male' or 'female') for current frame.
            gender_confidence: Estimation confidence of gender.
            
        Returns:
            A tuple of (aggregated_age, aggregated_gender, aggregated_gender_confidence, age_range).
        """
        with self._lock:
            now = time.time()
            if track_id not in self._history:
                self._history[track_id] = {
                    "window": deque(maxlen=settings.AGGREGATION_WINDOW),
                    "last_updated": now
                }
                
            history_entry = self._history[track_id]
            history_entry["window"].append({
                "age": age,
                "gender": gender,
                "gender_confidence": gender_confidence
            })
            history_entry["last_updated"] = now
            
            window_samples = list(history_entry["window"])

        # Calculations are done outside the mutex if possible, 
        # but since they are extremely fast O(W) with W <= 30, we can compute them easily.
        # 1. Average Age
        mean_age = float(np.mean([s["age"] for s in window_samples]))
        # Round age to nearest integer
        agg_age = int(round(mean_age))
        age_range = self.get_age_range(mean_age)
        
        # 2. Majority Vote for Gender
        genders = [s["gender"] for s in window_samples]
        male_count = genders.count("male")
        female_count = genders.count("female")
        
        if male_count >= female_count:
            agg_gender = "male"
            # Mean confidence of winning gender
            winning_samples = [s["gender_confidence"] for s in window_samples if s["gender"] == "male"]
        else:
            agg_gender = "female"
            winning_samples = [s["gender_confidence"] for s in window_samples if s["gender"] == "female"]
            
        agg_gender_confidence = float(np.mean(winning_samples)) if winning_samples else gender_confidence
        
        return agg_age, agg_gender, agg_gender_confidence, age_range

    def _cleanup_loop(self) -> None:
        """Background thread executing the memory garbage collector at regular intervals."""
        while True:
            time.sleep(60)  # Check every 60 seconds
            try:
                self.cleanup_inactive()
            except Exception as e:
                logger.error("Error during temporal aggregation service garbage collection", error=str(e))

    def cleanup_inactive(self) -> None:
        """Purges tracks that haven't been updated for longer than TRACKER_TTL."""
        with self._lock:
            now = time.time()
            expired_tracks = []
            for track_id, entry in self._history.items():
                if now - entry["last_updated"] > settings.TRACKER_TTL:
                    expired_tracks.append(track_id)
                    
            for track_id in expired_tracks:
                del self._history[track_id]
                logger.info(
                    "Evicted inactive facial track from aggregation service to free memory", 
                    track_id=track_id, 
                    ttl=settings.TRACKER_TTL
                )


# Instancia única Singleton global
aggregation_service = TemporalAggregationService()

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.core.config import settings
from src.core.logging import logger
from src.detectors.scrfd_detector import SCRFDDetector
from src.trackers.bytetrack_manager import tracker_registry, ByteTracker, STrack
from src.services.face_service import FaceService
from src.models.base import AgeGenderModel
from src.services.aggregation_service import aggregation_service


def _build_age_gender_model() -> AgeGenderModel:
    """
    Factory that instantiates the correct age/gender model based on MODEL_TYPE config.
    Supported values:
      - 'mivolo'      : HuggingFace MiVOLO V2 (ONNX if exported, otherwise PyTorch)
      - 'insightface' : InsightFace buffalo_l genderage.onnx (fast, pure ONNX)
    """
    model_type = settings.MODEL_TYPE.lower()

    if model_type == "insightface":
        from src.models.insightface_model import InsightFaceAgeGenderModel
        return InsightFaceAgeGenderModel(settings.INSIGHTFACE_MODEL_PATH)

    # Default: MiVOLO (with ONNX Runtime acceleration if mivolo_v2.onnx is present)
    from src.models.mivolo_model import MiVOLOModel
    return MiVOLOModel(settings.AGE_GENDER_MODEL_PATH)


class AgeGenderPipeline:
    """Consolidated pipeline for face detection, tracking, alignment, estimation, and temporal aggregation."""
    
    def __init__(self):
        self.detector = SCRFDDetector(settings.DETECTOR_MODEL_PATH)
        self.age_gender_model = _build_age_gender_model()

        # InsightFace requires 96x96 crops; MiVOLO uses 224x224.
        model_type = settings.MODEL_TYPE.lower()
        self._crop_size = 96 if model_type == "insightface" else settings.CROP_SIZE

        self.face_service = FaceService()
        logger.info(
            "AgeGenderPipeline initialized with selected modules",
            detector=settings.DETECTOR,
            model_type=settings.MODEL_TYPE,
            crop_size=self._crop_size
        )

    @staticmethod
    def _is_anatomically_valid_face(landmarks: np.ndarray, bbox: Optional[np.ndarray] = None) -> bool:
        """
        Validates if 5 facial landmarks conform to human face anatomy.
        Also performs a "face density" check using eye/mouth distances relative to
        bounding box dimensions to detect and reject extremely bloated boxes (body-sized detections).
        Landmarks layout:
          0: Left Eye (viewer's left, so smaller X)
          1: Right Eye (viewer's right, so larger X)
          2: Nose
          3: Left Mouth Corner
          4: Right Mouth Corner
        """
        if landmarks is None or landmarks.shape != (5, 2):
            return True # Fallback if no landmarks are available (e.g. Haar Cascade)
            
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # 1. Left eye must be to the left of the right eye
        if left_eye[0] >= right_eye[0]:
            return False
            
        # 2. Left mouth corner must be to the left of the right mouth corner
        if left_mouth[0] >= right_mouth[0]:
            return False
            
        # 3. Eyes must be above the mouth (y is 0 at top, so eyes Y must be smaller than mouth Y)
        if left_eye[1] >= left_mouth[1] or right_eye[1] >= right_mouth[1]:
            return False
            
        # 4. Nose must be vertically between eyes and mouth
        eyes_y_max = max(left_eye[1], right_eye[1])
        mouth_y_min = min(left_mouth[1], right_mouth[1])
        if nose[1] <= eyes_y_max or nose[1] >= mouth_y_min:
            return False
            
        # 5. Face Density / Proportions Check (only if bbox is provided)
        # Rejects "bloated" bounding boxes that contain shoulders, hats, or entire bodies.
        if bbox is not None:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bw > 0 and bh > 0:
                # Eye distance relative to bbox width (should be >= 20% in a tightly cropped face)
                eye_dist = right_eye[0] - left_eye[0]
                if eye_dist / bw < 0.20:
                    return False
                    
                # Eye-to-mouth Y distance relative to bbox height (should be >= 20%)
                eyes_y_avg = (left_eye[1] + right_eye[1]) / 2.0
                mouth_y_avg = (left_mouth[1] + right_mouth[1]) / 2.0
                eye_to_mouth_y = mouth_y_avg - eyes_y_avg
                if eye_to_mouth_y / bh < 0.20:
                    return False
            
        return True

    @staticmethod
    def _compute_iou_boxes(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Computes Intersect-over-Union (IoU) between two bounding boxes."""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)
        
        inter_w = max(0.0, xi2 - xi1)
        inter_h = max(0.0, yi2 - yi1)
        inter_area = inter_w * inter_h
        
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def process_frame(self, image: np.ndarray, camera_id: str) -> Dict[str, Any]:
        """
        Synchronously processes a video frame through the pipeline.
        
        Args:
            image: OpenCV image in BGR format.
            camera_id: Source camera identifier.
            
        Returns:
            Structured API dictionary response with tracking and demographics.
        """
        start_time = time.time()
        
        # 1. Face Detection (SCRFD)
        h_img, w_img = image.shape[:2]
        detections = self.detector.detect(image, threshold=0.5)

        # Filter out detections that are too large to be a face (body-sized or background false positives)
        # We allow very large close-up faces (up to 85% of the image) only if they are anatomically valid faces.
        # Otherwise, for normal sizes (<= 35% of the image), we accept them directly.
        MAX_FACE_RATIO = 0.85
        SAFE_FACE_RATIO = 0.35
        det_boxes = []
        det_scores = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bw = x2 - x1
            bh = y2 - y1
            
            # If it exceeds the absolute max ratio, reject
            if bw > w_img * MAX_FACE_RATIO or bh > h_img * MAX_FACE_RATIO:
                continue
                
            # If it's a large detection (> 35%), strictly validate its facial anatomy to reject background pareidolia
            if bw > w_img * SAFE_FACE_RATIO or bh > h_img * SAFE_FACE_RATIO:
                if not self._is_anatomically_valid_face(det.landmarks, det.bbox):
                    continue
                    
            det_boxes.append(det.bbox)
            det_scores.append(det.confidence)
            
        det_boxes_arr = np.array(det_boxes, dtype=np.float32) if det_boxes else np.empty((0, 4), dtype=np.float32)
        det_scores_arr = np.array(det_scores, dtype=np.float32) if det_scores else np.empty(0, dtype=np.float32)
        
        # 2. Get Camera Tracker and update it
        tracker: ByteTracker = tracker_registry.get_tracker(camera_id)
        active_tracks: List[STrack] = tracker.update(det_boxes_arr, det_scores_arr)
        
        faces_response = []
        
        # 3. Process each active track
        for track in active_tracks:
            track_bbox = track.tlbr
            
            # Find the best matching original detection to inherit landmarks
            best_iou = 0.0
            matched_detection = None
            
            for det in detections:
                iou = self._compute_iou_boxes(track_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    matched_detection = det
            
            # If IoU overlap is strong, we inherit original landmarks for precise alignment
            # Note: InsightFace genderage performs best on standard bounding box crops (without tight landmark warp)
            model_type = settings.MODEL_TYPE.lower()
            landmarks = matched_detection.landmarks if (matched_detection and best_iou > 0.5 and model_type != "insightface") else None
            
            # 4. Face Alignment and Cropping (crop size varies by model: 112 for InsightFace, 224 for MiVOLO)
            face_crop = self.face_service.align_and_crop(
                image, 
                bbox=track_bbox, 
                landmarks=landmarks,
                target_size=self._crop_size
            )
            
            # 5. Age and Gender Frame Estimation (MiVOLO)
            raw_result = self.age_gender_model.estimate(face_crop)
            
            # 6. Temporal Aggregation (Sliding Window)
            agg_age, agg_gender, agg_gender_confidence, age_range = aggregation_service.update_and_aggregate(
                track_id=track.track_id,
                age=raw_result.age,
                gender=raw_result.gender,
                gender_confidence=raw_result.gender_confidence
            )
            
            # Prepare face output dictionary
            # Include Bounding Box coordinates for client-side drawing and debugging
            faces_response.append({
                "track_id": track.track_id,
                "gender": agg_gender,
                "gender_confidence": round(agg_gender_confidence, 2),
                "age": agg_age,
                "age_range": age_range,
                "confidence": round(track.score, 2),
                "bbox": [int(track_bbox[0]), int(track_bbox[1]), int(track_bbox[2]), int(track_bbox[3])]
            })
            
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "camera_id": camera_id,
            "processing_time_ms": processing_time_ms,
            "faces": faces_response
        }

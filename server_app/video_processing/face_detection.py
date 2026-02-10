"""
Модуль детекции лиц
"""

from collections import deque
from time import time

import cv2
import mediapipe as mp

# Настройка MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Импорт модулей анализа (опциональные)
try:
    from .analyze_ear import EyeAspectRatioAnalyzer, classify_attention_by_ear
    from .analyze_head_pose import HeadPoseEstimator, classify_attention_state

    EAR_AVAILABLE = True
except ImportError:
    EAR_AVAILABLE = False
    print("Модули analyze_ear и/или analyze_head_pose не найдены. EAR и HeadPose анализ будет недоступен.")


class FaceDetector:
    """Модуль детекции лиц с использованием MediaPipe"""

    def __init__(self, *, min_detection_confidence: float = 0.5, margin: int = 20):
        """

        :param min_detection_confidence: Минимальный уровень уверенности модели, чтобы считать, что лицо есть
        :param margin: Добавочный отступ к bbox, предсказанный моделью
        """
        self._validate_margin(margin)
        self._validate_confidence(min_detection_confidence)

        self.detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range model (до 5 метров)
            min_detection_confidence=min_detection_confidence
        )
        self.margin = margin

    @staticmethod
    def _validate_margin(margin: int) -> None:
        """Валидация margin"""
        if not isinstance(margin, int):
            raise TypeError(f'Type of "margin" should be int, got {type(margin).__name__}')
        if margin < 0:
            raise ValueError('"margin" should be >= 0')

    def set_margin(self, margin: int):
        self._validate_margin(margin)
        self.margin = margin

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        """Валидация min_detection_confidence"""
        if not isinstance(confidence, (float, int)):
            raise TypeError(f'Type of confidence should be float, got {type(confidence).__name__}')
        if not 0 <= confidence <= 1:
            raise ValueError(f'Confidence should be in [0, 1], got {confidence}')

    def set_min_detection_confidence(self, min_detection_confidence: float):
        self._validate_confidence(min_detection_confidence)
        self.detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range model (до 5 метров)
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, image: cv2.typing.MatLike) -> list[dict[str,
    tuple[int, int, int, int] | cv2.typing.MatLike | float | list[tuple[int, int]]]]:
        """Детектирует лица на изображении"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                x1 = max(0, x - self.margin)
                y1 = max(0, y - self.margin)
                x2 = min(w, x + w_box + self.margin)
                y2 = min(h, y + h_box + self.margin)

                face_crop = image[y1:y2, x1:x2]

                # ИЗВЛЕКАЕМ КЛЮЧЕВЫЕ ТОЧКИ (6 точек)
                keypoints = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        keypoints.append((kp_x, kp_y))

                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'crop': face_crop,
                    'confidence': detection.score[0],
                    'keypoints': keypoints
                })

        return faces

    def close(self):
        self.detector.close()


if __name__ == '__main__':
    from video_processing.video_stream import process_video_stream

    print('Using camera 0')
    cap = cv2.VideoCapture(0)
    fps_history = deque()
    FPS_HISTORY_LEN = 3  # для более гладкого fps, будет выводится средние из последних FPS_HISTORY_LEN измерений

    for _ in range(FPS_HISTORY_LEN):
        fps_history.append(0.0)
    try:
        start_time = time()
        for img, emotions in process_video_stream(cap, flip_h=True):
            cv2.putText(img, f'FPS: {round(sum(fps_history) / FPS_HISTORY_LEN)}', (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.imshow('Test', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break  # esc to quit
            fps = 1 / (time() - start_time)
            fps_history.append(fps)
            fps_history.popleft()
            start_time = time()
    finally:
        print('Releasing resources...')
        cap.release()
        cv2.destroyAllWindows()
        print('Done!')

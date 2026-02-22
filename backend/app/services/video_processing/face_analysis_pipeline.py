from dataclasses import dataclass

import cv2
import torch

from .analyze_ear import EyeAspectRatioAnalyzer, EyeAspectRatioAnalyzeResult
from .analyze_emotion import EmotionRecognizer
from .analyze_head_pose import HeadPoseEstimator, HeadPoseEstimatResult
from .face_detection import FaceDetector, mp_face_mesh


@dataclass
class OneFaceMetricsAnalizResult:
    emotion: str
    confidence: float
    bbox: tuple[int, int, int, int]
    ear: EyeAspectRatioAnalyzeResult | None
    head_pose: HeadPoseEstimatResult | None


@dataclass
class FaceAnalizResult:
    image: cv2.typing.MatLike
    metrics: list[OneFaceMetricsAnalizResult]


class FaceAnalysisPipeline:
    """Пайплайн для комплексного анализа лица (детекция + эмоции + EAR + HeadPose)"""

    def __init__(self, face_detector: FaceDetector, emotion_recognizer: EmotionRecognizer,
                 ear_analyzer: EyeAspectRatioAnalyzer | None = None,
                 head_pose_estimator: HeadPoseEstimator | None = None, use_face_mesh: bool = True):
        """
        Args:
            face_detector: Детектор лиц
            emotion_recognizer: Распознаватель эмоций
            ear_analyzer: EyeAspectRatioAnalyzer (опционально)
            head_pose_estimator: HeadPoseEstimator (опционально)
            use_face_mesh: Использовать Face Mesh для EAR/HeadPose (требует больше ресурсов)
        """
        self.face_detector = face_detector
        self.emotion_recognizer = emotion_recognizer
        self.ear_analyzer = ear_analyzer
        self.head_pose_estimator = head_pose_estimator
        self.use_face_mesh = use_face_mesh

        # Инициализируем Face Mesh, если нужен для EAR или HeadPose
        self.face_mesh = None
        if use_face_mesh and (ear_analyzer or head_pose_estimator):
            self._init_face_mesh()

    def _init_face_mesh(self):
        """Инициализирует Face Mesh для EAR/HeadPose анализа"""
        if self.face_mesh is None:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def _close_face_mesh(self):
        """Закрывает Face Mesh"""
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def set_ear_analyzer(self, ear_analyzer):
        """Устанавливает или сбрасывает EAR анализатор (hot-reload)"""
        self.ear_analyzer = ear_analyzer
        # Инициализируем Face Mesh, если нужен
        if ear_analyzer or self.head_pose_estimator:
            self._init_face_mesh()
        elif not self.head_pose_estimator:
            self._close_face_mesh()

    def set_head_pose_estimator(self, head_pose_estimator):
        """Устанавливает или сбрасывает HeadPose анализатор (hot-reload)"""
        self.head_pose_estimator = head_pose_estimator
        # Инициализируем Face Mesh, если нужен
        if head_pose_estimator or self.ear_analyzer:
            self._init_face_mesh()
        elif not self.ear_analyzer:
            self._close_face_mesh()

    def analyze(self, image: cv2.typing.MatLike) -> FaceAnalizResult:
        """
        Детектирует лица и распознаёт эмоции (Опционально - EAR и HeadPose)
        :param image: Входное изображение с лицами для анализа
        :return: Возвращает изображение с bbox'ами и список результатов для каждого лица.
        Формат: (image, [{'emotion': str, 'confidence': float, 'ear': dict, 'head_pose': dict}, ...])
        """
        vis_image = image.copy()  # создание копии для отрисовки на ней bbox'ов
        faces = self.face_detector.detect(image)

        # Запускаем Face Mesh, если нужен
        face_mesh_results = None
        if self.face_mesh:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_mesh_results = self.face_mesh.process(rgb_image)

        results = []
        h, w, _ = image.shape

        for face_idx, face in enumerate(faces):
            # 1. Распознавание эмоции
            prediction = self.emotion_recognizer.predict(face.crop)
            emotion = prediction.label
            conf = prediction.confidence

            x1, y1, x2, y2 = face.bbox

            # 2. EAR анализ (если доступен Face Mesh)
            if self.ear_analyzer and face_mesh_results and face_mesh_results.multi_face_landmarks:
                if face_idx < len(face_mesh_results.multi_face_landmarks):
                    face_landmarks = face_mesh_results.multi_face_landmarks[face_idx]
                    ear_result = self.ear_analyzer.analyze(face_landmarks, w, h, face_idx)
                    ear = ear_result
                else:
                    ear = None
            else:
                ear = None

            # 3. HeadPose анализ (если доступен Face Mesh)
            if self.head_pose_estimator and face_mesh_results and face_mesh_results.multi_face_landmarks:
                if face_idx < len(face_mesh_results.multi_face_landmarks):
                    face_landmarks = face_mesh_results.multi_face_landmarks[face_idx]
                    head_pose_result = self.head_pose_estimator.estimate(face_landmarks, w, h)
                    head_pose = head_pose_result
                else:
                    head_pose = None
            else:
                head_pose = None

            result = OneFaceMetricsAnalizResult(emotion=emotion, confidence=conf, bbox=(x1, y1, x2, y2),
                                                ear=ear, head_pose=head_pose)

            # 4. Визуализация
            self._draw_face_info(vis_image, result)

            results.append(result)

        return FaceAnalizResult(vis_image, results)

    @staticmethod
    def _draw_face_info(image: cv2.typing.MatLike, result: OneFaceMetricsAnalizResult):
        """Отрисовывает информацию о лице на изображении"""
        x1, y1, x2, y2 = result.bbox

        # Рисуем bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Эмоция
        emotion_text = f"{result.emotion}: {result.confidence:.2f}"
        cv2.putText(image, emotion_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # EAR (если доступен)
        y_offset = y1 - 30
        if result.ear:
            ear_text = f"EAR: {result.ear.avg_ear:.3f}"
            cv2.putText(image, ear_text, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset -= 15

            if result.ear.is_blinking:
                blink_text = "BLINK"
                cv2.putText(image, blink_text, (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_offset -= 15

        # HeadPose (если доступен)
        if result.head_pose:
            pitch = result.head_pose.pitch
            yaw = result.head_pose.yaw
            roll = result.head_pose.roll
            head_pose_text = f"P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}"
            cv2.putText(image, head_pose_text, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def make_face_analysis_pipeline() -> FaceAnalysisPipeline:
    return FaceAnalysisPipeline(face_detector=FaceDetector(),
                                emotion_recognizer=EmotionRecognizer(
                                    device='cuda' if torch.cuda.is_available() else 'cpu'),
                                ear_analyzer=EyeAspectRatioAnalyzer(), head_pose_estimator=HeadPoseEstimator())

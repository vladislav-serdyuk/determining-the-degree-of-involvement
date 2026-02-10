import cv2

from .analyze_ear import EyeAspectRatioAnalyzer, classify_attention_by_ear
from .analyze_emotion import EmotionRecognizer
from .analyze_head_pose import HeadPoseEstimator, classify_attention_state
from .face_detection import FaceDetector, EAR_AVAILABLE, mp_face_mesh


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

    def analyze(self, image: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, list[dict]]:
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
            emotion, conf = self.emotion_recognizer.predict(face['crop'])
            x1, y1, x2, y2 = face['bbox']

            result = {
                'emotion': emotion,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            }

            # 2. EAR анализ (если доступен Face Mesh)
            if self.ear_analyzer and face_mesh_results and face_mesh_results.multi_face_landmarks:
                if face_idx < len(face_mesh_results.multi_face_landmarks):
                    face_landmarks = face_mesh_results.multi_face_landmarks[face_idx]
                    ear_result = self.ear_analyzer.analyze(face_landmarks, w, h, face_idx)
                    if ear_result and EAR_AVAILABLE:
                        # Добавляем классификацию состояния внимания по EAR
                        ear_result['attention_state'] = classify_attention_by_ear(
                            ear_result['avg_ear'],
                            ear_result['blink_count']
                        )
                    result['ear'] = ear_result
                else:
                    result['ear'] = None
            else:
                result['ear'] = None

            # 3. HeadPose анализ (если доступен Face Mesh)
            if self.head_pose_estimator and face_mesh_results and face_mesh_results.multi_face_landmarks:
                if face_idx < len(face_mesh_results.multi_face_landmarks):
                    face_landmarks = face_mesh_results.multi_face_landmarks[face_idx]
                    head_pose_result = self.head_pose_estimator.estimate(face_landmarks, w, h)
                    if head_pose_result and EAR_AVAILABLE:
                        # Добавляем классификацию состояния внимания
                        head_pose_result['attention_state'] = classify_attention_state(
                            head_pose_result['pitch'],
                            head_pose_result['yaw'],
                            head_pose_result['roll']
                        )
                    result['head_pose'] = head_pose_result
                else:
                    result['head_pose'] = None
            else:
                result['head_pose'] = None

            # 4. Визуализация
            self._draw_face_info(vis_image, result)

            results.append(result)

        return vis_image, results

    def _draw_face_info(self, image: cv2.typing.MatLike, result: dict):
        """Отрисовывает информацию о лице на изображении"""
        x1, y1, x2, y2 = result['bbox']

        # Рисуем bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Эмоция
        emotion_text = f"{result['emotion']}: {result['confidence']:.2f}"
        cv2.putText(image, emotion_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # EAR (если доступен)
        y_offset = y1 - 30
        if result['ear']:
            ear_text = f"EAR: {result['ear']['avg_ear']:.3f}"
            cv2.putText(image, ear_text, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset -= 15

            if result['ear']['is_blinking']:
                blink_text = "BLINK"
                cv2.putText(image, blink_text, (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_offset -= 15

        # HeadPose (если доступен)
        if result['head_pose']:
            pitch = result['head_pose']['pitch']
            yaw = result['head_pose']['yaw']
            roll = result['head_pose']['roll']
            head_pose_text = f"P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}"
            cv2.putText(image, head_pose_text, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

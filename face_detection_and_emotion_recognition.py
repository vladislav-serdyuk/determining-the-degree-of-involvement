"""
Модуль детекции лиц и распознавания эмоций
"""

import typing
from time import time
from collections import deque

import torch
import cv2
import mediapipe as mp
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from emotiefflib.facial_analysis import get_model_list

# Настройка MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Импорт модулей анализа (опциональные)
try:
    from analyze_ear import EyeAspectRatioAnalyzer, classify_attention_by_ear
    from analyze_head_pose import HeadPoseEstimator, classify_attention_state
    EAR_AVAILABLE = True
except ImportError:
    EAR_AVAILABLE = False
    print("Модули analyze_ear и/или analyze_head_pose не найдены. EAR и HeadPose анализ будет недоступен.")


class FaceDetector:
    """Детектор лиц MediaPipe Full-Range (для разных дистанций)"""

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


class EmotionRecognizer:
    """Распознавание с temporal smoothing + confidence thresholding"""

    def __init__(self, *, device: typing.Literal['cpu', 'cuda'] = 'cpu', window_size=15,
                 confidence_threshold=0.55, ambiguity_threshold=0.15,
                 model_name='enet_b2_8'):
        """
        Args:
            device: 'cpu' или 'cuda'. Не меняется после инициализации
            window_size: Размер окна для сглаживания
            confidence_threshold: Минимальный порог уверенности
            ambiguity_threshold: Порог для амбивалентных эмоций
            model_name: Имя модели
        """
        if not isinstance(device, str):
            raise TypeError(f'Type of "device" should be str, got {type(device).__name__}')
        if device not in ['cpu', 'cuda']:
            raise ValueError(f'"device" should be "cpu" or "cuda". Got "{device}"')
        self._validate_window_size(window_size)
        self._validate_confidence_threshold(confidence_threshold)
        self._validate_ambiguity_threshold(ambiguity_threshold)
        if not isinstance(model_name, str):
            raise TypeError(f'Type of "model_name" should be str, got {type(model_name).__name__}')
        if model_name not in get_model_list():
            raise ValueError(f'Unknown "model_name". Got "{model_name}". Available models: {get_model_list()}')

        self.recognizer = EmotiEffLibRecognizer(
            model_name=model_name,
            device=device
        )
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy',
                               'Sad', 'Surprise', 'Neutral', 'Contempt']

        # Параметры сглаживания
        self.history = deque(maxlen=window_size)

        # Параметры фильтрации
        self.confidence_threshold = confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold
        print(f"EmotiEffLib + Advanced загружен: модель={model_name}, устройство={device}")

    @staticmethod
    def _validate_window_size(window_size: int):
        if not isinstance(window_size, int):
            raise TypeError(f'Type of "window_size" should be int, got {type(window_size).__name__}')
        if window_size < 0:
            raise ValueError(f'"window_size" should be >= 0')

    def set_window_size(self, window_size: int):
        self._validate_window_size(window_size)
        self.history = deque(maxlen=window_size)

    @staticmethod
    def _validate_confidence_threshold(confidence_threshold: float):
        if not isinstance(confidence_threshold, (float, int)):
            raise TypeError(f'Type of "confidence_threshold" should be float, got {type(confidence_threshold).__name__}')
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError(f'"confidence_threshold" should be in [0;1]')

    def set_confidence_threshold(self, confidence_threshold: float):
        self._validate_confidence_threshold(confidence_threshold)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _validate_ambiguity_threshold(ambiguity_threshold: float):
        if not isinstance(ambiguity_threshold, (float, int)):
            raise TypeError(f'Type of "ambiguity_threshold" should be float, got {type(ambiguity_threshold).__name__}')
        if ambiguity_threshold < 0 or ambiguity_threshold > 1:
            raise ValueError(f'"ambiguity_threshold" should be in [0;1]')

    def set_ambiguity_threshold(self, ambiguity_threshold: float):
        self._validate_ambiguity_threshold(ambiguity_threshold)
        self.ambiguity_threshold = ambiguity_threshold

    def predict(self, face_crop: cv2.typing.MatLike) -> tuple[str, float]:
        """Предсказывает эмоцию с продвинутой фильтрацией"""
        if face_crop.size == 0:
            return "Neutral", 0.0  # Fallback к нейтральному

        try:
            # Получаем предсказание
            emotion, scores = self.recognizer.predict_emotions(face_crop, logits=True)

            # Берём топ эмоцию и confidence
            top_emotion = emotion[0]

            if scores is not None and len(scores) > 0:
                confidence = float(max(scores[0])) if hasattr(scores[0], '__iter__') else float(scores[0])
            else:
                confidence = 1.0

            # Шаг 1: Проверка confidence threshold
            if confidence < self.confidence_threshold:
                # Слишком низкая уверенность -> нейтральное состояние
                top_emotion = "Neutral"
                confidence = self.confidence_threshold * 0.9

            # Добавляем в историю
            self.history.append({
                'emotion': top_emotion,
                'confidence': confidence
            })

            # Шаг 2: Temporal smoothing
            if len(self.history) >= 3:
                emotion_votes = {}
                total_weight = 0

                for i, hist_item in enumerate(self.history):
                    weight = (i + 1) / len(self.history)
                    emo = hist_item['emotion']
                    conf = hist_item['confidence']

                    if emo not in emotion_votes:
                        emotion_votes[emo] = 0
                    emotion_votes[emo] += weight * conf
                    total_weight += weight

                # Сортируем эмоции по весу
                sorted_emotions = sorted(
                    emotion_votes.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Шаг 3: Проверка амбивалентности
                if len(sorted_emotions) >= 2:
                    top_emotion_result, top_score = sorted_emotions[0]
                    second_emotion, second_score = sorted_emotions[1]

                    # Если две топ-эмоции слишком близки -> нейтральное
                    if (top_score - second_score) / total_weight < self.ambiguity_threshold:
                        return "Neutral", 0.5

                    return top_emotion_result, top_score / total_weight
                else:
                    top_emotion_result, top_score = sorted_emotions[0]
                    return top_emotion_result, top_score / total_weight

            return top_emotion, confidence

        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Критично - пробрасываем выше для обработки
            print('Out of memory in EmotionRecognizer.predict()')
            raise

        except (ValueError, RuntimeError, AttributeError) as e:
            # Ожидаемые проблемы обработки - логируем и fallback
            print(f"Предупреждение при распознавании: {e}")
            return "Neutral", 0.0

    def reset(self):
        """Сброс истории"""
        self.history.clear()


class DetectFaceAndRecognizeEmotion:
    """Детектирует лица и распознаёт эмоции (с опциональной доп. интеграцией EAR и HeadPose)"""

    def __init__(self, face_detector: FaceDetector, emotion_recognizer: EmotionRecognizer,
                 ear_analyzer=None, head_pose_estimator=None, use_face_mesh: bool = True):
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

    def detect_and_recognize(self, image: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, list[dict]]:
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


class CaptureReadError(Exception): pass


def process_video_stream(video_stream: cv2.VideoCapture,
                         face_detector_and_emotion_recognizer: DetectFaceAndRecognizeEmotion | None = None, *,
                         flip_h: bool = False):
    """
    Обрабатывает видеопоток, находя лица и распознавая эмоции
    :param video_stream: Видео поток
    :param face_detector_and_emotion_recognizer: То, с помощью чего обрабатывается видеопоток
    :param flip_h: Отзеркалить входящий видеопоток
    :return: Генератор, который возвращает обработанный кадр и эмоции. Формат: (image, [(emotion, confidence), ...])
    """
    use_inner_models = face_detector_and_emotion_recognizer is None
    if use_inner_models:
        face_detector = FaceDetector(min_detection_confidence=0.5)
        emotion_recognizer = EmotionRecognizer(device='cuda' if torch.cuda.is_available() else 'cpu', window_size=15,
                                               confidence_threshold=0.55, ambiguity_threshold=0.15)
        face_detector_and_emotion_recognizer = DetectFaceAndRecognizeEmotion(face_detector, emotion_recognizer)

    if not video_stream.isOpened():
        raise CaptureReadError('"video_stream" is not opened')
    try:
        while True:
            ret_val, img = video_stream.read()
            if not ret_val:
                raise CaptureReadError('Failed to get image from "video_stream"')
            if flip_h:
                img = cv2.flip(img, 1)
            new_img, emotions = face_detector_and_emotion_recognizer.detect_and_recognize(img)
            yield new_img, emotions
    finally:
        if use_inner_models:
            face_detector.close()
            emotion_recognizer.reset()
            del face_detector_and_emotion_recognizer


if __name__ == '__main__':
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

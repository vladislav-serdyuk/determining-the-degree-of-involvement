from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import torch
from emotiefflib.facial_analysis import get_model_list, EmotiEffLibRecognizer


@dataclass
class EmotionRecognizeResult:
    label: str
    confidence: float


class EmotionRecognizer:
    """Распознавание с temporal smoothing + confidence thresholding"""

    def __init__(self, *, device: Literal['cpu', 'cuda'] = 'cpu', window_size=15,
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
            raise TypeError(
                f'Type of "confidence_threshold" should be float, got {type(confidence_threshold).__name__}')
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

    def predict(self, face_crop: cv2.typing.MatLike) -> EmotionRecognizeResult:
        """Предсказывает эмоцию с продвинутой фильтрацией"""
        if face_crop.size == 0:
            return EmotionRecognizeResult('Neutral', 0.0)  # Fallback к нейтральному

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
                        return EmotionRecognizeResult('Neutral', 0.5)

                    return EmotionRecognizeResult(top_emotion_result, top_score / total_weight)
                else:
                    top_emotion_result, top_score = sorted_emotions[0]
                    return EmotionRecognizeResult(top_emotion_result, top_score / total_weight)

            return EmotionRecognizeResult(top_emotion, confidence)

        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Критично - пробрасываем выше для обработки
            print('Out of memory in EmotionRecognizer.predict()')
            raise

        except (ValueError, RuntimeError, AttributeError) as e:
            # Ожидаемые проблемы обработки - логируем и fallback
            print(f"Предупреждение при распознавании: {e}")
            return EmotionRecognizeResult('Neutral', 0.0)

    def reset(self):
        """Сброс истории"""
        self.history.clear()

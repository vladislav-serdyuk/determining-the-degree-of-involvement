from collections import deque
from dataclasses import dataclass
from typing import cast

import cv2
import torch
from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # type: ignore[import-untyped]

from app.core.config import settings


@dataclass
class EmotionRecognizeResult:
    label: str
    confidence: float


class EmotionRecognizer:
    """Распознавание с temporal smoothing + confidence thresholding"""

    device = "cuda" if settings.emotion_device == "auto" else settings.emotion_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    recognizer = EmotiEffLibRecognizer(model_name=settings.emotion_model_name, device=device)
    print(f"EmotiEffLib + Advanced загружен: модель={settings.emotion_model_name}, устройство={device}")

    def __init__(self, *, window_size=None, confidence_threshold=None, ambiguity_threshold=None):
        """
        Args:
            window_size: Размер окна для сглаживания
            confidence_threshold: Минимальный порог уверенности
            ambiguity_threshold: Порог для амбивалентных эмоций
        """
        actual_window_size = window_size if window_size is not None else settings.emotion_window_size
        actual_confidence = (
            confidence_threshold if confidence_threshold is not None else settings.emotion_confidence_threshold
        )
        actual_ambiguity = (
            ambiguity_threshold if ambiguity_threshold is not None else settings.emotion_ambiguity_threshold
        )

        self._validate_window_size(actual_window_size)
        self._validate_confidence_threshold(actual_confidence)
        self._validate_ambiguity_threshold(actual_ambiguity)

        self.emotion_labels = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
            "Contempt",
        ]

        # Параметры сглаживания
        self.history: deque[dict[str, float | str]] = deque(maxlen=actual_window_size)

        # Параметры фильтрации
        self.confidence_threshold = actual_confidence
        self.ambiguity_threshold = actual_ambiguity

    @staticmethod
    def _validate_window_size(window_size: int):
        if not isinstance(window_size, int):
            raise TypeError(f'Type of "window_size" should be int, got {type(window_size).__name__}')
        if window_size < 0:
            raise ValueError('"window_size" should be >= 0')

    def set_window_size(self, window_size: int):
        self._validate_window_size(window_size)
        self.history = deque(maxlen=window_size)

    @staticmethod
    def _validate_confidence_threshold(confidence_threshold: float):
        if not isinstance(confidence_threshold, (float, int)):
            raise TypeError(
                f'Type of "confidence_threshold" should be float, got {type(confidence_threshold).__name__}'
            )
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError('"confidence_threshold" should be in [0;1]')

    def set_confidence_threshold(self, confidence_threshold: float):
        self._validate_confidence_threshold(confidence_threshold)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _validate_ambiguity_threshold(ambiguity_threshold: float):
        if not isinstance(ambiguity_threshold, (float, int)):
            raise TypeError(f'Type of "ambiguity_threshold" should be float, got {type(ambiguity_threshold).__name__}')
        if ambiguity_threshold < 0 or ambiguity_threshold > 1:
            raise ValueError('"ambiguity_threshold" should be in [0;1]')

    def set_ambiguity_threshold(self, ambiguity_threshold: float):
        self._validate_ambiguity_threshold(ambiguity_threshold)
        self.ambiguity_threshold = ambiguity_threshold

    def predict(self, face_crop: cv2.typing.MatLike) -> EmotionRecognizeResult:
        """Предсказывает эмоцию с продвинутой фильтрацией"""
        if face_crop.size == 0:
            return EmotionRecognizeResult("Neutral", 0.0)  # Fallback к нейтральному

        try:
            # Получаем предсказание
            emotion, scores = self.recognizer.predict_emotions(face_crop, logits=True)

            # Берём топ эмоцию и confidence
            top_emotion = emotion[0]

            if scores is not None and len(scores) > 0:
                confidence = float(max(scores[0])) if hasattr(scores[0], "__iter__") else float(scores[0])
            else:
                confidence = 1.0

            # Шаг 1: Проверка confidence threshold
            if confidence < self.confidence_threshold:
                # Слишком низкая уверенность -> нейтральное состояние
                top_emotion = "Neutral"
                confidence = self.confidence_threshold * 0.9

            # Добавляем в историю
            self.history.append({"emotion": top_emotion, "confidence": confidence})

            # Шаг 2: Temporal smoothing
            if len(self.history) >= 3:
                emotion_votes: dict[str, float] = {}
                total_weight: float = 0.0

                for i, hist_item in enumerate(self.history):
                    weight = (i + 1) / len(self.history)
                    emo = cast(str, hist_item["emotion"])
                    conf = cast(float, hist_item["confidence"])

                    if emo not in emotion_votes:
                        emotion_votes[emo] = 0.0
                    emotion_votes[emo] += weight * conf
                    total_weight += weight

                # Сортируем эмоции по весу
                sorted_emotions = sorted(emotion_votes.items(), key=lambda x: x[1], reverse=True)

                # Шаг 3: Проверка амбивалентности
                if len(sorted_emotions) >= 2:
                    top_emotion_result, top_score = sorted_emotions[0]
                    second_emotion, second_score = sorted_emotions[1]

                    # Если две топ-эмоции слишком близки -> нейтральное
                    if (top_score - second_score) / total_weight < self.ambiguity_threshold:
                        return EmotionRecognizeResult("Neutral", 0.5)

                    return EmotionRecognizeResult(top_emotion_result, top_score / total_weight)
                else:
                    top_emotion_result, top_score = sorted_emotions[0]
                    return EmotionRecognizeResult(top_emotion_result, top_score / total_weight)

            return EmotionRecognizeResult(top_emotion, confidence)

        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Критично - пробрасываем выше для обработки
            print("Out of memory in EmotionRecognizer.predict()")
            raise

        except (ValueError, RuntimeError, AttributeError) as e:
            # Ожидаемые проблемы обработки - логируем и fallback
            print(f"Предупреждение при распознавании: {e}")
            return EmotionRecognizeResult("Neutral", 0.0)

    def reset(self):
        """Сброс истории"""
        self.history.clear()

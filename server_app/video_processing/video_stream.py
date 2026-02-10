import cv2
import torch

from .analyze_emotion import EmotionRecognizer
from .face_analysis_pipeline import FaceAnalysisPipeline
from .face_detection import FaceDetector


class CaptureReadError(Exception): pass


def process_video_stream(video_stream: cv2.VideoCapture,
                         face_analyze_pipeline: FaceAnalysisPipeline | None = None, *,
                         flip_h: bool = False):
    """
    Обрабатывает видеопоток, находя лица и распознавая эмоции
    :param video_stream: Видео поток
    :param face_analyze_pipeline: То, с помощью чего обрабатывается видеопоток
    :param flip_h: Отзеркалить входящий видеопоток
    :return: Генератор, который возвращает обработанный кадр и эмоции. Формат: (image, [(emotion, confidence), ...])
    """
    use_inner_models = face_analyze_pipeline is None
    if use_inner_models:
        face_detector = FaceDetector(min_detection_confidence=0.5)
        emotion_recognizer = EmotionRecognizer(device='cuda' if torch.cuda.is_available() else 'cpu', window_size=15,
                                               confidence_threshold=0.55, ambiguity_threshold=0.15)
        face_analyze_pipeline = FaceAnalysisPipeline(face_detector, emotion_recognizer)

    if not video_stream.isOpened():
        raise CaptureReadError('"video_stream" is not opened')
    try:
        while True:
            ret_val, img = video_stream.read()
            if not ret_val:
                raise CaptureReadError('Failed to get image from "video_stream"')
            if flip_h:
                img = cv2.flip(img, 1)
            new_img, emotions = face_analyze_pipeline.analyze(img)
            yield new_img, emotions
    finally:
        if use_inner_models:
            face_detector.close()
            emotion_recognizer.reset()
            del face_analyze_pipeline

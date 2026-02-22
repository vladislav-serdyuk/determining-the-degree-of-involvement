import torch

from services.video_processing import FaceDetector, EmotionRecognizer, EyeAspectRatioAnalyzer, HeadPoseEstimator, \
    FaceAnalysisPipeline

models = {}


def load_models():
    print("Загрузка моделей...")
    models['face_detector'] = FaceDetector()
    models['emotion_recognizer'] = EmotionRecognizer(device='cuda' if torch.cuda.is_available() else 'cpu')
    models['ear_analyzer'] = EyeAspectRatioAnalyzer()
    models['head_analyzer'] = HeadPoseEstimator()
    models['analyzer'] = FaceAnalysisPipeline(
        models['face_detector'],
        models['emotion_recognizer'],
        models['ear_analyzer'],
        models['head_analyzer']
    )
    print("Модели загружены")


def close_models():
    print("Закрытие моделей")
    models.clear()

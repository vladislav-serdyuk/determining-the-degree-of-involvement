# ML-пайплайн

Раздел документации, охватывающий пайплайн обработки от получения кадра с веб-камеры, то вычисления значений эмоций, HPE, EAR. Охватывает модули компьютерного зрения и применения моделей машинного обучения, отвечающих за анализ видеопотока пользователя, связь с модулями EAR и HPE.

Оркестрация происходит через  [`FaceAnalysisPipeline`](../../backend/app/services/video_processing/face_analysis_pipeline.py)

---

## Содержание

1. [Обзор пайплайна](pipeline-overview.md) - общая схема, порядок вызовов, ресурсы
2. [Детекция лиц](face-detection.md) - MediaPipe Face Detection
3. [Распознавание эмоций](emotion-recognition.md) - EmotiEffLib (EfficientNet-B2)
4. [Анализ состояния глаз (EAR)](eye-aspect-ratio.md) - Eye Aspect Ratio, моргания, attention_state
5. Поза головы (HPE) - Perspective-n-Point (PnP), углы Эйлера, attention_state
6. Временное сглаживание - как сигнал стабилизируется на каждом уровне



---

## Схема потока

```
Кадр (BGR)
    │
    ├── FaceDetector.detect()                            ── bbox, crop, keypoints
    │       │
    │       └──► EmotionRecognizer.predict(crop)         ── emotion, confidence
    │
    └── FaceMesh.process()  [если включены EAR или HPE]
            │
            ├──► EyeAspectRatioAnalyzer.analyze(lm, w, h, face_id)  ── EAR, blinks, attention_state
            └──► HeadPoseEstimator.estimate(lm, w, h)               ── pitch/yaw/roll, attention_state
                                                                              │
                                                            EngagementCalculator.calculate()
                                                                              │
                                                                              ▼
                                                                      score, level, trend
```



## Опциональность модулей

EAR и HPE являются опциональными (отключаемыми). Если оба модуля отключены, то Face Mesh не инициализируется, экономятся ресурсы. См. [face_analysis_pipeline.py:68-72](../../backend/app/services/video_processing/face_analysis_pipeline.py#L68-L72). Параметр `use_face_mesh` в конструкторе `FaceAnalysisPipeline` автоматически управляется через [`make_face_analysis_pipeline`](../../backend/app/services/video_processing/face_analysis_pipeline.py#L247).

При отсутствии данных одного из модулей калькулятор вовлечённости `EngagementCalculator` возвращает нейтральное значение `0.5` для соответствующего компонента.


## Визуализация на кадре

`FaceAnalysisPipeline._draw_face_info` накладывает аннотации поверх копии входного кадра:

| Элемент | Цвет (BGR) | Формат |
|---------|-----------|--------|
| Bbox | Magenta `(255, 0, 255)` | Прямоугольник |
| Эмоция | Magenta `(255, 0, 255)` | `Happiness: 0.87` |
| EAR | Бирюзовый `(0, 255, 200)` | `EAR: 0.355 [Alert]` |
| BLINK | Красный `(0, 0, 255)` | текст `BLINK` при `is_blinking` |
| Head Pose | Голубой `(255, 200, 0)` | `P:5 Y:-8 R:2 [Attentive]` |
| Engagement | Зелёный `(0, 255, 0)` | `Eng: 0.78 (High)` |


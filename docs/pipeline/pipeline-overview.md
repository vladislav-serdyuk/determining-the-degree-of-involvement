# Оркестратор `FaceAnalysisPipeline`

**Файл:** [`backend/app/services/video_processing/face_analysis_pipeline.py`](../../backend/app/services/video_processing/face_analysis_pipeline.py)

Класс объединяет все модули анализа в единый вызов `analyze(image) → FaceAnalyzeResult`.


## Входные параметры конструктора

| Параметр | Тип | Обязательный | Назначение |
|----------|-----|--------------|-----------|
| `face_detector` | `FaceDetector` | да | Детектор лиц MediaPipe |
| `emotion_recognizer` | `EmotionRecognizer` | да | Модель EmotiEffLib |
| `ear_analyzer` | `EyeAspectRatioAnalyzer \| None` | нет | EAR-анализ глаз |
| `head_pose_estimator` | `HeadPoseEstimator \| None` | нет | PnP-оценка позы головы |
| `engagement_calculator` | `EngagementCalculator \| None` | нет | Расчёт итогового score |
| `use_face_mesh` | `bool` | default `True` | Подтягивает MediaPipe Face Mesh, если нужен хотя бы одному из EAR/HPE |

Если `ear_analyzer` и `head_pose_estimator` оба `None`, то Face Mesh не инициализируется независимо от `use_face_mesh`.

*Обязательность* модулей описана в [README.md](README.md#опциональность-модулей). Под необязательностью модуля `EngagementCalculator` понимаем то, что можно получать собранные метрики без использования представленного модуля вычисления итоговой вовлечённости. Например, если нужны сами показатели, а не конечная метрика. 

## Порядок обработки одного кадра

Реализация в `FaceAnalysisPipeline.analyze`:

| Шаг | Действие |
|-----|----------|
| 1 | Копирование кадра в `vis_image` для рисования |
| 2 | `FaceDetector.detect(image)` → список `FaceDetectResult` | 
| 3 | BGR→RGB и `face_mesh.process(rgb_image)` (если активен) | 
| 4 | Для каждого лица: `emotion_recognizer.predict(face.crop)` |
| 5 | Для каждого лица: `ear_analyzer.analyze(landmarks, w, h, face_idx)` |
| 6 | Для каждого лица: `head_pose_estimator.estimate(landmarks, w, h)` |
| 7 | Для каждого лица: `engagement_calculator.calculate(...)` (если активен) |
| 8 | `_draw_face_info(vis_image, result)` - аннотации на кадре | 
| 9 | Возврат `FaceAnalyzeResult(vis_image, metrics)` | 


## Сопоставление лиц между детектором и Face Mesh

Face Detection и Face Mesh - **независимые** модели MediaPipe. Пайплайн сопоставляет лица по индексу в списке: `face_idx`-е лицо от `FaceDetector` сопоставляется с `face_idx`-м ландмарком из `face_mesh_results.multi_face_landmarks`.

```python
# face_analysis_pipeline.py:
if face_idx < len(face_mesh_results.multi_face_landmarks):
    face_landmarks = face_mesh_results.multi_face_landmarks[face_idx]
    ear_result = self.ear_analyzer.analyze(face_landmarks, w, h, face_idx)
```

**Ограничение:** порядок лиц в двух моделях может различаться при нескольких лицах в кадре. Система рассчитана на single-face сценарий (один пользователь перед камерой). При `face_mesh_max_num_faces > 1` возможны кросс-привязки landmarks не к тому лицу.


## Обработка ошибок

Все вызовы ML-моделей обёрнуты в `try/except`:

| Источник ошибки | Поведение |
|-----------------|-----------|
| `FaceDetector.detect` | `faces = []`, лог `error` |
| `face_mesh.process` | `face_mesh_results = None`, лог `error` |
| `EmotionRecognizer.predict` | `emotion="unknown", conf=0.0` | 
| `ear_analyzer.analyze` | `ear = None` | 
| `head_pose_estimator.estimate` | `head_pose = None` | 
| `engagement_calculator.calculate` | `engagement = None` |

Частичный сбой одного модуля не прерывает пайплайн: в WS-ответе соответствующее поле будет возвращено как `null`.


## Выходные данные

```python
@dataclass
class OneFaceMetricsAnalyzeResult:
    emotion: str
    confidence: float
    bbox: tuple[int, int, int, int]
    ear: EyeAspectRatioAnalyzeResult | None
    head_pose: HeadPoseEstimateResult | None
    engagement: EngagementCalculateResult | None

@dataclass
class FaceAnalyzeResult:
    image: cv2.typing.MatLike         # кадр с аннотациями
    metrics: list[OneFaceMetricsAnalyzeResult]
```

Формат сериализации в WS-ответе описан в [api-format.md](../api-format.md).

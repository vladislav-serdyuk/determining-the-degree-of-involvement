# Детекция лиц

**Файл:** [`backend/app/services/video_processing/face_detection.py`](../../backend/app/services/video_processing/face_detection.py)
**Модель:** MediaPipe Face Detection


## Вход

OpenCV-изображение BGR (`cv2.typing.MatLike`). Разрешение не регламентируется, но MediaPipe работает быстрее на изображениях до 640x480.

> Использование BGR (blue, green, red) вместо RGB связано с тем, что данный формат является нативным для OpenCV, и не влияет на сохраняемую внутри цветовую информацию.


## Выход

`list[FaceDetectResult]` - по одному элементу на каждое обнаруженное лицо.

```python
@dataclass
class FaceDetectResult:
    bbox: tuple[int, int, int, int]        # (x1, y1, x2, y2) в пикселях
    crop: cv2.typing.MatLike               # вырезанная область лица (BGR)
    confidence: float                      # уверенность детекции [0.0, 1.0]
    keypoints: list[tuple[int, int]]       # 6 ключевых точек: глаза, нос, рот, уши
```

Пустой список - если лиц не найдено.


## Предобработка

1. **BGR → RGB** через `cv2.cvtColor`. MediaPipe принимает RGB.
2. **Нормализованные координаты → пиксельные.** MediaPipe возвращает `relative_bounding_box` в долях ширины/высоты; код пересчитывает через `int(bbox.xmin * w)` и т. п.
3. **Добавление margin и клиппинг к границам кадра** - для того, чтобы модель эмоций получила чуть более широкий crop с контекстом вокруг лица (важно, например, если crop'ом не были захвачены уши или подбородок):

   ```python
   x1 = max(0, x - self.margin)
   y1 = max(0, y - self.margin)
   x2 = min(w, x + w_box + self.margin)
   y2 = min(h, y + h_box + self.margin)
   ```

4. **Вырезка crop** `image[y1:y2, x1:x2]` - передаётся дальше в `EmotionRecognizer`.


## Параметры

| Атрибут класса | Env-переменная | По умолчанию | Диапазон | Описание |
|----------------|----------------|--------------|----------|----------|
| `model_selection` | `FACE_DETECTION_MODEL_SELECTION` | `1` | `{0, 1}` | `0` - short-range (до 2 м), `1` - full-range (до 5 м) |
| `min_detection_confidence` | `FACE_DETECTION_MIN_CONFIDENCE` | `0.5` | `[0.0, 1.0]` | Минимальная уверенность модели для принятия детекции |
| `margin` | `FACE_DETECTION_MARGIN` | `20` | `>= 0` | Отступ в пикселях вокруг bbox |

Все три параметра читаются из `settings` ([config.py](../../backend/app/core/config.py)) при создании экземпляра. `min_detection_confidence` и `margin` можно переопределить в конструкторе:

```python
FaceDetector(min_detection_confidence=0.7, margin=30)
```

---

## Runtime-изменение параметров

| Метод | Что делает | Особенность |
|-------|-----------|-------------|
| `set_margin(margin)` | Меняет отступ | Без пересоздания модели |
| `set_min_detection_confidence(conf)` | Меняет порог уверенности | **Пересоздаёт** `mp_face_detection.FaceDetection` - возможна кратковременная пауза |

Оба метода валидируют вход: `margin` - неотрицательный int, `confidence` - float в `[0, 1]`.

---

## Ключевые точки (`keypoints`)

MediaPipe Face Detection возвращает 6 keypoints (порядок фиксирован):

| Индекс | Точка |
|--------|-------|
| 0 | Правый глаз |
| 1 | Левый глаз |
| 2 | Кончик носа |
| 3 | Центр рта |
| 4 | Правое ухо (tragion) |
| 5 | Левое ухо (tragion) |

Не стоит путать с 468 landmarks из Face Mesh (используются в EAR и HPE).

Keypoints включены в `FaceDetectResult`, но в WS-ответе не передаются - используются только для внутреннего дебага.

---

## Закрытие ресурсов

MediaPipe выделяет нативные ресурсы, их нужно освобождать при остановке:

```python
detector.close()      # закрывает mp_face_detection.FaceDetection
```

В продакшен-пути закрытие не требуется - `FaceDetector` живёт в синглтоне пайплайна на всё время работы backend. В `tools/param_testing_app.py` закрытие выполняется при явной остановке захвата камеры.

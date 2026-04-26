# Распознавание эмоций

**Файл:** [`backend/app/services/video_processing/analyze_emotion.py`](../../backend/app/services/video_processing/analyze_emotion.py)
**Модель:** EmotiEffLib (`enet_b2_8` - EfficientNet-B2, 8 классов AffectNet)


## Вход

Crop лица в формате BGR (`cv2.typing.MatLike`). Используется результат `FaceDetector.detect`, поле `crop`. При передаче пустого массива (`size == 0`) происходит fallback `EmotionRecognizeResult("Neutral", 0.0)`.

> Использование BGR (blue, green, red) вместо RGB связано с тем, что данный формат является нативным для OpenCV, и не влияет на сохраняемую внутри цветовую информацию.


## Выход

```python
@dataclass
class EmotionRecognizeResult:
    label: str          # одно из 8 значений, см. ниже
    confidence: float   # нормированный вес после сглаживания [0.0, 1.0]
```

> `confidence` в результате - это **не** вероятность модели в строгом смысле. После temporal smoothing это `top_score / total_weight`, то есть нормированная сумма "взвешенных голосов" по топ-эмоции в окне. Если в результате взвешенного голосования топ 2 эмоции имеют разницу в значении ниже, чем `ambiguity_threshold`, то считаем, что они компенсируют друг друга ("одинаковые значения"), тогда возвращается `("Neutral", 0.5)`; при confidence ниже порога `confidence_threshold` до сглаживания - `("Neutral", confidence_threshold * 0.9)`.


## Классы эмоций (AffectNet, 8 классов)

| Индекс | Label | Типовая интерпретация |
|--------|-------|----------------------|
| 0 | `Anger` | Гнев, фрустрация |
| 1 | `Contempt` | Скептицизм, презрение |
| 2 | `Disgust` | Отвращение |
| 3 | `Fear` | Страх, тревога |
| 4 | `Happiness` | Радость, улыбка |
| 5 | `Neutral` | Спокойное состояние |
| 6 | `Sadness` | Грусть, усталость |
| 7 | `Surprise` | Удивление, интерес |

Точные метки классов (лейблы) берутся из `EmotiEffLibRecognizer.idx_to_emotion_class`. Если модель вернёт лейбл вне этого списка, `EngagementCalculator` выбросит `KeyError` на этапе `EMOTION_WEIGHTS[emotion]`.


## Конвейер обработки эмоций (3 этапа)

### Этап 1. Confidence thresholding

Если уверенность топ-эмоции от модели ниже `confidence_threshold` (default `0.55`), предсказание заменяется на `Neutral` с искусственно подавленной уверенностью:

```python
# analyze_emotion.py:175-178
if confidence < self.confidence_threshold:
    top_emotion = "Neutral"
    confidence = self.confidence_threshold * 0.9  # для default: 0.495
```

Нужно для фильтрации шумных предсказаний при плохом освещении или нетипичном ракурсе.

### Этап 2. Temporal smoothing (взвешенное голосование)

Активируется при `len(history) >= 3`. Каждому кадру из окна присваивается линейно возрастающий вес, свежие кадры значимее:

```python
# analyze_emotion.py:184-196
for i, hist_item in enumerate(self.history):
    weight = (i + 1) / len(self.history)
    emotion_votes[emo] += weight * conf
    total_weight += weight
```

Для каждой эмоции суммируется `weight * confidence` по всем кадрам окна, где она предсказывалась. Побеждает эмоция с максимальной суммой голосов.

Эффекты:
- Устойчивость к одиночным ложным предсказаниям.
- Приоритет недавних кадров (линейно растущий вес).
- Учёт уверенности в голосе: `conf=0.9` вносит в 3 раза больший вклад, чем `conf=0.3`.

### Этап 3. Ambiguity filtering

Если разница первой и второй эмоций по нормированному голосу меньше `ambiguity_threshold` (default `0.15`), то считаем, что эмоции имеют одинаковый вес и компенсируют друг друга - возвращаем `("Neutral", 0.5)`.

```python
# analyze_emotion.py:207-208
if (top_score - second_score) / total_weight < self.ambiguity_threshold:
    return EmotionRecognizeResult("Neutral", 0.5)
```

Предотвращает "мерцание" между двумя близкими эмоциями в пограничных ситуациях (например, `Happiness` vs `Surprise`).

Если в окне не нашлось двух эмоций (например, вся история содержит только один класс), то ambiguity-проверка пропускается.


## Параметры

| Параметр | Env-переменная | По умолчанию | Описание |
|----------|----------------|--------------|----------|
| `model_name` | `EMOTION_MODEL_NAME` | `enet_b2_8` | Имя модели из EmotiEffLib |
| `device` | `EMOTION_DEVICE` | `auto` | `cpu`, `cuda`, `auto` (определяется по `torch.cuda.is_available()`) |
| `window_size` | `EMOTION_WINDOW_SIZE` | `15` | Размер `deque` истории |
| `confidence_threshold` | `EMOTION_CONFIDENCE_THRESHOLD` | `0.55` | Порог confidence thresholding |
| `ambiguity_threshold` | `EMOTION_AMBIGUITY_THRESHOLD` | `0.15` | Порог ambiguity filtering |


### Выбор устройства

```
device=auto  - cuda если доступен, иначе cpu
device=cuda  - cuda если доступен; иначе выберется cpu с предупреждением
device=cpu   - cpu
```

Переопределение в `__init__(device=...)` передаёт выбор обратно через `settings.emotion_device` - локальное значение параметра игнорируется. Настоящий выбор диктуется env-переменной.

## Runtime-изменение параметров

| Метод | Эффект |
|-------|--------|
| `set_window_size(n)` | Пересоздаёт `deque(maxlen=n)` - **история очищается** |
| `set_confidence_threshold(x)` | Меняет порог, история не затрагивается |
| `set_ambiguity_threshold(x)` | Меняет порог, история не затрагивается |
| `reset()` | Очищает историю |

Используется в инструменте `frontend/tools/param_testing_app.py` для изменения параметров в ходе тестирования.

## Обработка ошибок

| Источник | Поведение |
|----------|-----------|
| `torch.cuda.OutOfMemoryError`, `MemoryError` | Пробрасывается наверх, логируется как `error` |
| `ValueError`, `RuntimeError`, `AttributeError` | Возвращается `("Neutral", 0.0)`, логируется как `warning` |
| Пустой `face_crop` (`size == 0`) | Возвращается `("Neutral", 0.0)` без вызова модели |

# Head Pose Estimation (HPE). Оценка положения головы в пространстве

**Файл:** [`backend/app/services/video_processing/analyze_head_pose.py`](../../backend/app/services/video_processing/analyze_head_pose.py)
**Метод:** Perspective-n-Point (PnP) на 6 ключевых лицевых точках, полученных от MediaPipe


## Вход

| Параметр | Тип | Описание |
|----------|-----|----------|
| `face_landmarks` | MediaPipe object | Ландмарки одного лица из Face Mesh |
| `image_width` | `int` | Ширина кадра (для pinhole-приближения `focal_length ≈ width`) |
| `image_height` | `int` | Высота кадра |


## Выход

```python
@dataclass
class HeadPoseEstimateResult:
    pitch: float                          # наклон вверх/вниз, градусы
    yaw: float                            # поворот влево/вправо, градусы
    roll: float                           # наклон к плечу, градусы
    rotation_vec: tuple[float, float, float]      # вектор Родригеса
    translation_vec: tuple[float, float, float]   # смещение
    attention_state: Literal["Highly Attentive", "Attentive", "Distracted", "Very Distracted"]
```

Возврат `None`, если `cv2.solvePnP` не нашёл решение.

### Соглашение о знаках

| Угол | Положительный знак | Типовое «внимание к экрану» |
|------|---------------------|------------------------------|
| `pitch` | голова наклонена вверх | около `0` |
| `yaw` | голова повёрнута вправо | около `0` |
| `roll` | голова наклонена к правому плечу | около `0` |

---

## Алгоритм

### 1. Извлечение 2D-точек

```python
HEAD_POSE_LANDMARKS = [1, 33, 61, 199, 263, 291]
```

| Индекс MediaPipe | Точка |
|------------------|-------|
| 1 | Nose tip (кончик носа) |
| 33 | Left eye outer corner (внешний угол левого глаза) |
| 61 | Left mouth corner (левый угол рта) |
| 199 | Chin (подбородок) |
| 263 | Right eye outer corner (внешний угол правого глаза) |
| 291 | Right mouth corner (правый угол рта) |

### 2. 3D-модель усреднённого лица

```python
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (-225.0, -170.0, 135.0), # Left eye corner
    (-150.0,  150.0, 125.0), # Left mouth corner
    (0.0, 330.0, 65.0),      # Chin
    (225.0, -170.0, 135.0),  # Right eye corner
    (150.0,  150.0, 125.0),  # Right mouth corner
])
```

Единицы условные; важно отношение координат, не абсолютная шкала.

### 3. Модель камеры (pinhole, без дисторшена)

```python
focal_length = image_width
center = (image_width / 2, image_height / 2)
camera_matrix = [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]]
dist_coeffs = zeros((4, 1))
```

Приближение `focal_length ≈ image_width` применимо для типовых веб-камер без калибровки. Искажения линзы игнорируются.

### 4. Решение PnP

```python
success, rotation_vec, translation_vec = cv2.solvePnP(
    MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE,
)
```

### 5. Вектор → матрица → углы Эйлера

```python
rotation_mat, _ = cv2.Rodrigues(rotation_vec)
angles = cv2.RQDecomp3x3(rotation_mat)[0]    # (pitch, yaw, roll) в градусах
pitch, yaw, roll = angles[0], angles[1], angles[2]
```

### 6. Коррекция roll

После `RQDecomp3x3` значение roll иногда скачет к ±180°. Реализована нормализация значения:

```python
if abs(roll) > 100:
    roll = roll + 180 if roll < 0 else roll - 180
```

После коррекции roll лежит в околонулевом диапазоне для стандартного вертикального положения головы.


## Классификация `attention_state`

Функция [`classify_attention_state(pitch, yaw, roll)`](../../backend/app/services/video_processing/analyze_head_pose.py#L160).

Проверки идут последовательно на `|pitch|` и `|yaw|`. Первая удовлетворяющая ветка возвращается:

```python
if |pitch| < 10 and |yaw| < 15:  → "Highly Attentive"
elif |pitch| < 20 and |yaw| < 25: → "Attentive"
elif |pitch| < 30 and |yaw| < 40: → "Distracted"
else:                              → "Very Distracted"
```

**Особенность:** `roll` в классификацию **не входит**. Наклон к плечу не уменьшает attention_state - при таком отклонении визуальный контакт с экраном сохраняется, делая это дополнительное условие избыточным. Учитываем только отклонения "влево-вправо", "вверх-вниз".

| `attention_state` | |pitch| | |yaw| | Смысл |
|-------------------|---------|-------|-------|
| `Highly Attentive` | < 10° | < 15° | Прямой взгляд на экран |
| `Attentive` | < 20° | < 25° | Лёгкое отклонение (клавиатура, тетрадь) |
| `Distracted` | < 30° | < 40° | Заметный поворот в сторону |
| `Very Distracted` | иначе | иначе | Взгляд сильно в сторону, вниз или вверх |

---

## Параметры классификации

| Порог | Env-переменная | По умолчанию |
|-------|----------------|--------------|
| `pitch` (highly) | `HEAD_PITCH_HIGHLY_ATTENTIVE` | `10.0` |
| `yaw` (highly) | `HEAD_YAW_HIGHLY_ATTENTIVE` | `15.0` |
| `pitch` (attentive) | `HEAD_PITCH_ATTENTIVE` | `20.0` |
| `yaw` (attentive) | `HEAD_YAW_ATTENTIVE` | `25.0` |
| `pitch` (distracted) | `HEAD_PITCH_DISTRACTED` | `30.0` |
| `yaw` (distracted) | `HEAD_YAW_DISTRACTED` | `40.0` |

Все пороги читаются из `settings` - меняются через `.env` при запуске.


## Связь с engagement

`attention_state` маппится в `head_pose_score` через `HEAD_POSE_STATE_SCORES` в `EngagementCalculator`. См. [../engagement-calculation/component-scores.md](../engagement-calculation/component-scores.md).


## Ограничения метода

- **Одна усреднённая 3D-модель лица** - для лиц с нетипичной геометрией (большие очки, борода, далеко посаженные глаза) могут быть смещения до нескольких градусов.
- **Отсутствие калибровки камеры.** `focal_length = image_width` - приблизительное усреднённое значение, что позволяет использовать HPE без предварительной калибровки; тем не менее, для камер с широким углом оценка yaw/pitch может быть смещена по абсолюту.

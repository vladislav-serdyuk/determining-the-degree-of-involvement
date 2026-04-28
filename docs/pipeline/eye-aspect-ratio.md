# Eye Aspect Ratio (EAR). Анализ состояния глаз

**Файл:** [`backend/app/services/video_processing/analyze_ear.py`](../../backend/app/services/video_processing/analyze_ear.py)

**Источник landmarks:** MediaPipe Face Mesh (468 точек, подмножество из 6 точек на каждый глаз)


## Вход

| Параметр | Тип | Описание |
|----------|-----|----------|
| `face_landmarks` | MediaPipe object | Ландмарки одного лица из `face_mesh_results.multi_face_landmarks[i]` |
| `image_width` | `int` | Ширина кадра (для денормализации) |
| `image_height` | `int` | Высота кадра |
| `face_id` | `int` | Идентификатор лица для отдельного хранения истории и blink-счётчиков (в текущей реализации совпадает с `face_idx` из пайплайна) |


## Выход

```python
@dataclass
class EyeAspectRatioAnalyzeResult:
    left_ear: float
    right_ear: float
    avg_ear: float
    eyes_open: bool                       # avg_ear >= ear_threshold
    blink_count: int                      # всего подтверждённых морганий в сессии
    is_blinking: bool                     # avg_ear < ear_threshold в текущем кадре
    ear_history: list[float] | None       # deque последних N значений (N = ear_history_maxlen)
    attention_state: Literal["Alert", "Normal", "Drowsy", "Very Drowsy"]
```

> Поле `ear_history` сериализуется в `result_to_dict` с удалением при публикации в Redis Pub/Sub для экономии трафика. В WS-ответах клиенту оно тоже возвращается как `null` (см. [api-format.md](../api-format.md)). Внутри бэкенда хранится в `deque(maxlen=settings.ear_history_maxlen)`.



## Формула EAR

По работе Soukupová & Čech (2016), *Real-Time Eye Blink Detection using Facial Landmarks* ([прямая ссылка на pdf-файл работы](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)):

```
EAR = (‖P₂−P₆‖ + ‖P₃−P₅‖) / (2 · ‖P₁−P₄‖)
```

Точки P₁…P₆ обходят контур глаза:
- P₁, P₄ - горизонтальные углы (внешний, внутренний)
- P₂, P₆ - первая вертикальная пара (верхнее/нижнее веко)
- P₃, P₅ - вторая вертикальная пара

Реализация: [`EyeAspectRatioAnalyzer._calculate_ear`](../../backend/app/services/video_processing/analyze_ear.py#L80-L104). 

> **Защита от деления на ноль**: при `C ≤ 0` возвращается `0.0`.

### Индексы Face Mesh

```python
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 145]
LEFT_EYE_LANDMARKS  = [362, 385, 387, 263, 373, 380]
```

Порядок соответствует `[P₁, P₂, P₃, P₄, P₅, P₆]`.

### Типовые значения

| Состояние | EAR |
|-----------|-----|
| Глаза широко открыты | 0.30–0.40 |
| Нормальное раскрытие | 0.25–0.30 |
| Прикрытые | 0.20–0.25 |
| Почти закрытые | < 0.20 |


## Детекция моргания

Цель: отфильтровать ложные срабатывания при шуме ландмарков. Моргание **подтверждается** только если глаза были закрыты `consec_frames` кадров подряд и затем открылись.

Алгоритм (для каждого лица, индексируется по `face_id`):

```
если avg_ear < ear_threshold:
    blink_counter[face_id] += 1
    is_blinking = True
иначе:
    если blink_counter[face_id] >= consec_frames:
        blink_totals[face_id] += 1     # подтверждённое моргание
    blink_counter[face_id] = 0
```

Источник: [`analyze_ear.py:147-158`](../../backend/app/services/video_processing/analyze_ear.py#L147-L158).

`blink_count` в результирующем dataclass - это `blink_totals[face_id]`, то есть **накопленное число** с начала сессии.

---

## Классификация `attention_state`

Функция [`classify_attention_by_ear(avg_ear, blink_rate)`](../../backend/app/services/video_processing/analyze_ear.py#L185).

```python
if avg_ear >= ear_alert_threshold (0.30):
    if 10 <= blink_rate <= 25:  → Alert
    else:                        → Normal
elif avg_ear >= ear_drowsy_threshold (0.20):
    → Normal
elif avg_ear >= ear_very_drowsy_threshold (0.15):
    → Drowsy
else:
    → Very Drowsy
```

Замысел вложенной развилки на верхней ветке `if`: при широко открытых глазах (`avg_ear >= 0.30`) **и** нормальной частоте морганий (10-25/мин по литературе - см. [Magliacano et al., 2020](https://doi.org/10.1016/j.neulet.2020.135293)) человек активно вовлечён → `Alert` (*внимателен*). Если частота вне нормы (слишком редко = гиперфокус/усталость; слишком часто = стресс/раздражение), то состояние остаётся `Normal` - глаза формально открыты, но дополнительного "бонуса" внимательности не присуждается.

> **Замечание**. В связи с описанной ниже особенностью реализации данная развилка **является избыточной, кандидатом на упрощение**, поскольку blink_rate в данном контексте - общее число морганий. Данная логика не нарушает работы пайплайна засчёт сглаживания итоговой метрики в `EngagementCalculator`, а описанная в [Magliacano et al., 2020](https://doi.org/10.1016/j.neulet.2020.135293) логика служит модификатором данной визуальной метрики на уровне вычисления вовлечённости.


> **Особенность реализации.** Параметр называется `blink_rate` и семантически подразумевается как **частота в минуту**, однако фактически `EyeAspectRatioAnalyzer.analyze` передаёт в него `self.blink_totals[face_id]` - **общее количество** подтверждённых морганий за сессию:
>
> ```python
> # analyze_ear.py:168
> attention_state=classify_attention_by_ear(avg_ear, self.blink_totals[face_id]),
> ```
>
> Следствие: на этапе `avg_ear ≥ 0.30` пока всего моргнули < 10 раз состояние будет `Normal`, затем, когда накопится 10–25 морганий - `Alert`, после 25+ - снова `Normal`. Эта логика **является тем, что исполняется сейчас**, и отличается от кажущейся из названия "за минуту". Полноценный blink-rate-бонус в минуту применяется **отдельно** в `EngagementCalculator.calculate_eye_score` через соответствующий модификатор.


| `attention_state` | EAR | blink_count | Интерпретация |
|-------------------|-----|-------------|---------------|
| `Alert` | ≥ 0.30 | 10–25 | Широко открытые глаза + стабильная активность моргания |
| `Normal` | ≥ 0.30, blink_count вне [10,25] | - | Открытые глаза, но старт сессии или высокий кумулятивный счёт |
| `Normal` | 0.20–0.30 | - | Пограничное раскрытие |
| `Drowsy` | 0.15–0.20 | - | Начало усталости |
| `Very Drowsy` | < 0.15 | - | Глаза почти закрыты |


## Параметры

| Параметр | Env-переменная | По умолчанию | Описание |
|----------|----------------|--------------|----------|
| `ear_threshold` | `EAR_THRESHOLD` | `0.25` | Порог закрытых глаз для `is_blinking` / `eyes_open` |
| `consec_frames` | `EAR_CONSEC_FRAMES` | `3` | Кадров подряд с закрытыми глазами, чтобы засчитать моргание |
| `history_maxlen` | `EAR_HISTORY_MAXLEN` | `30` | Размер `deque` истории EAR (`~1 сек` при 30 FPS) |
| `ear_alert_threshold` | `EAR_ALERT_THRESHOLD` | `0.30` | Порог `Alert` |
| `ear_drowsy_threshold` | `EAR_DROWSY_THRESHOLD` | `0.20` | Граница `Normal` vs `Drowsy` |
| `ear_very_drowsy_threshold` | `EAR_VERY_DROWSY_THRESHOLD` | `0.15` | Граница `Drowsy` vs `Very Drowsy` |


## Runtime-изменение и сброс

| Метод | Эффект |
|-------|--------|
| `set_ear_threshold(x)` | Меняет порог, **не сбрасывает** счётчики моргания |
| `set_consec_frames(n)` | Меняет минимум кадров, не сбрасывает счётчики |
| `reset()` | Очищает все словари (`blink_counters`, `blink_totals`, `ear_history`) для всех лиц |
| `reset(face_id=i)` | Очищает данные только для лица `i` |

Пороги для `classify_attention_by_ear` (`ear_alert/drowsy/very_drowsy_threshold`) живут в `settings` и не имеют setter-методов - меняются только перезапуском при изменении в `.env`.


## Связь с engagement (вовлечённостью)

`attention_state` напрямую маппится в `eye_score` через `EAR_STATE_SCORES` в `EngagementCalculator`. Дополнительный модификатор по частоте моргания в минуту вычисляется отдельно.

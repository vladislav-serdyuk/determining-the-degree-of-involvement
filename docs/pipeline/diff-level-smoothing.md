# Временное сглаживание в пайплайне

Сигналы от ML-моделей зашумлены: даже стабильный взгляд даёт некоторые флуктуации эмоций, EAR и углов головы. 

Система применяет **независимое сглаживание на разных уровнях** *(ML-пайплайн, калькулятор вовлечённости)*, чтобы итоговые метрики были устойчивыми, но при этом быстро реагировали на реальные изменения.


## Где происходит сглаживание

| Уровень | Файл | Что сглаживается | Окно |
|---------|------|-------------------|------|
| Эмоции | [`analyze_emotion.py`](../../backend/app/services/video_processing/analyze_emotion.py) | Label + confidence | `window_size = 15` |
| Engagement | [`engagement_calculator.py`](../../backend/app/services/video_processing/engagement_calculator.py) | Итоговый score | Адаптивное 15 или 45 |
| Engagement trend | [`engagement_calculator.py`](../../backend/app/services/video_processing/engagement_calculator.py) | Тенденция | `trend_window = 30` |

EAR и HPE **не сглаживаются внутри своих модулей (!)** - каждый кадр независим. Их сглаживание выполняется при вычислении итогового показателя `engagement`.

---

## Уровни сглаживания

Далее приводится подробный алгоритм сглаживания на каждом из уровней.

---

### Уровень эмоций - взвешенное голосование

Подробнее в [emotion-recognition.md](emotion-recognition.md#этап-2-temporal-smoothing-взвешенное-голосование).

Алгоритм:

1. Заполняется `deque(maxlen=15)` парами `{emotion, confidence}`.
2. Каждому кадру присваивается линейно возрастающий вес: `weight_i = (i+1)/len(history)`.
3. Для каждой эмоции в окне суммируется `weight \* confidence`.
4. Побеждает эмоция с максимальной суммой. Её score нормируется по `total_weight` (общему накопленному весу).
5. Если перевес первой над второй меньше `ambiguity_threshold` - возврат `Neutral` (защита от мерцания, эмоции *"компенсировались"*).

Зачем такая схема:
- Линейный вес даёт быстрый отклик на новые эмоции (последние 3–4 кадра доминируют).
- Умножение на confidence - не пускает неуверенные предсказания, не позволяя им портить голосование.
- Ambiguity-фильтр - явно ловит пограничные ситуации, где модель колеблется между двумя классами.

---

### Уровень engagement

#### Engagement. Адаптивное окно

Реализация в [`EngagementCalculator.calculate`](../../backend/app/services/video_processing/engagement_calculator.py#L260-L274).

```python
if len(history) < 5:
    smoothed = raw                              # стартовая фаза: пока < 5 кадров, сглаживание выключено
else:
    recent = history[-15:]                      # последние ~0.5 сек
    if np.var(recent) < bypass_threshold:
        smoothed = np.mean(recent)              # стабильно - короткое окно (быстрая реакция)
    else:
        smoothed = np.mean(history)             # изменчиво - полное окно (сильное сглаживание, до 45)
```

| Параметр | По умолчанию | Смысл |
|----------|--------------|-------|
| `window_size` | `45` | Максимум окна (~1.5 сек при 30 FPS) |
| `bypass_threshold` | `0.08` | Если дисперсия последних 15 значений меньше - считаем сигнал стабильным |
| `trend_window` | `30` | Отдельное окно для тренда (см. ниже) |

Логика:
- **Стабильное состояние** (учащийся спокойно смотрит в экран) - окно `15`, среднее быстро сдвигается вверх/вниз при настоящем изменении.
- **Шумное состояние** (много движения, ложные детекции) - окно `45`, агрессивное усреднение, меньше всплесков.

Нижний порог "прогрева" (стартовая фаза) в 5 кадров предотвращает выдачу сглаженного нуля в первые кадры сессии.


#### Engagement. Тренд

Реализация: [`EngagementCalculator._calculate_trend`](../../backend/app/services/video_processing/engagement_calculator.py#L318-L343).

**Цель** - дать оценку (grade): `rising` / `falling` / `stable`, не пересчитывая производную.

Алгоритм:

```python
if len(trend_history) < 10:
    return "stable"

half = len(trend_history) // 2
diff = mean(trend_history[half:]) - mean(trend_history[:half])

if diff >  0.05: return "rising"
if diff < -0.05: return "falling"
return "stable"
```

То есть сравниваются средние первой и второй половины окна 30 кадров. Порог `0.05` защищает от ложных тенденций/срабатываний при локальных колебаниях.

При `< 10` значениях (старт сессии) всегда `stable`.

---

## Взаимодействие уровней

```
Кадры → эмоции сглаживаются (w.voting, 15)
                  │
                  ▼
           emotion_score (raw каждый кадр)
                  │
                  │   eye_score (raw)
                  │   head_pose_score (raw)
                  ▼
        Weighted sum → engagement_raw
                  │
                  ▼
     adaptive window → engagement score (smoothed)
                  │
                  ▼
        trend_window (30) → trend
```

Эмоциональное сглаживание уменьшает дрожание `emotion_score` - что, в свою очередь, уменьшает вариативность `engagement_raw`, что помогает адаптивному окну чаще попадать в "стабильный" режим.

---

## Сброс

| Метод | Что очищает |
|-------|-------------|
| `EmotionRecognizer.reset()` | `self.history` (deque эмоций) |
| `EngagementCalculator.reset()` | `engagement_history`, `trend_history`, `session_start_time`, счётчики |
| `EyeAspectRatioAnalyzer.reset()` | `blink_counters`, `blink_totals`, `ear_history` (нужны в контексте смены пользователя) |

Вызываются при старте новой сессии или явной замене параметров (например, при `set_window_size`).

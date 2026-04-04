# API Формат данных

> OpenAPI спецификация: [openapi.yaml](openapi.yaml)

## REST API

### Health Check

**GET** `/health`

```json
Response:
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Список комнат

**GET** `/rooms`

```json
Response: ["room_id_1", "room_id_2", ...]
```

### Клиенты в комнате

**GET** `/rooms/{room_id}/clients`

```json
Response:
[
  { "name": "client_name", "id_": "uuid-строка" },
  ...
]
```

---

## WebSocket API

### Эндпоинт анализа видеопотока

**URL**: `ws://localhost:8000/ws/rooms/{room_id}/stream?name={client_name}`

#### Отправка (клиент → сервер)

```python
# Python-клиент (пример)
import json
import base64

data = {
    "image": "base64_encoded_jpeg_image"
}
ws.send(json.dumps(data))
```

#### Принятие (сервер → клиент)

```python
# Успешный ответ
{
    "image": "base64_encoded_processed_jpeg",  # Изображение с наложениями
    "results": [
        {
            "emotion": "Happiness",
            "confidence": 0.85,
            "bbox": [x1, y1, x2, y2],
            "ear": {
                "left_ear": 0.28,
                "right_ear": 0.27,
                "avg_ear": 0.275,
                "eyes_open": true,
                "blink_count": 5,
                "is_blinking": false,
                "ear_history": [0.28, 0.27, ...],
                "attention_state": "Normal"
            },
            "head_pose": {
                "pitch": 5.2,
                "yaw": -2.1,
                "roll": 1.0,
                "rotation_vec": [0.1, 0.2, 0.05],
                "translation_vec": [0, 0, 50],
                "attention_state": "Highly Attentive"
            },
            "engagement": {
                "score": 0.75,      # Сглаженный (адаптивное сглаживание)
                "score_raw": 0.82,  # Сырой (текущий кадр)
                "level": "High",
                "trend": "stable",
                "components": {
                    "emotion_score": 0.8,
                    "eye_score": 0.7,
                    "head_pose_score": 0.75
                },
                "frame_count": 150
            }
        }
    ]
}

# Ошибка
{
    "error": "Failed to decode image: ..."
}
```

---

## Типы данных (Pydantic)

### Входящие

```python
class FrameRequest(BaseModel):
    image: str  # base64 JPEG
```

### Исходящие

```python
class EARResult(BaseModel):
    left_ear: float
    right_ear: float
    avg_ear: float
    eyes_open: bool
    blink_count: int
    is_blinking: bool
    ear_history: list[float] | None = None
    attention_state: Literal["Alert", "Normal", "Drowsy", "Very Drowsy"]

class HeadPoseResult(BaseModel):
    pitch: float          # градусы
    yaw: float            # градусы
    roll: float           # градусы
    rotation_vec: tuple[float, float, float]
    translation_vec: tuple[float, float, float]
    attention_state: Literal["Highly Attentive", "Attentive", "Distracted", "Very Distracted"]

class EngagementComponents(BaseModel):
    emotion_score: float
    eye_score: float
    head_pose_score: float

class EngagementResult(BaseModel):
    score: float          # 0.0 - 1.0 (сглаженный)
    score_raw: float       # 0.0 - 1.0 (сырой)
    level: Literal["High", "Medium", "Low", "Very Low"]
    trend: Literal["rising", "falling", "stable"]
    components: EngagementComponents
    frame_count: int

class FaceAnalysisResult(BaseModel):
    emotion: str  # Happiness, Neutral, Sadness и т.д.
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    ear: EARResult | None
    head_pose: HeadPoseResult | None
    engagement: EngagementResult | None

class FrameResponse(BaseModel):
    image: str             # base64 обработанного JPEG
    results: list[FaceAnalysisResult]

class OutputStreamFrameResponse(BaseModel):
    image_src: str         # base64 исходного JPEG
    image: str             # base64 обработанного JPEG
    results: list[FaceAnalysisResult]

class ErrorResponse(BaseModel):
    error: str
```

---

## WebSocket для просмотра потока другого клиента

**URL**: `ws://localhost:8000/ws/rooms/{room_id}/clients/{client_id}/output_stream`

Отдаёт:
```json
{
    "image_src": "base64...",
    "image": "base64...",
    "results": [...]
}
```

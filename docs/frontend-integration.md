# Подключение фронтенда

## Обзор

Фронтенд подключается к бэкенду через **WebSocket** для отправки видеокадров и получения результатов анализа.

## Варианты интеграции

### 1. api_client (существующий)

```python
from api_client import EngagementAPIClient

# Инициализация
client = EngagementAPIClient(
    backend_ws_url="ws://localhost:8000",
    backend_http_url="http://localhost:8000"
)

# Подключение
client.connect(room_id="my-room", name="user-name")

# Отправка кадра (OpenCV BGR numpy array)
frame = cv2.imread("image.jpg")
processed_frame, results = client.send_frame(frame)
# processed_frame - обработанное изображение BGR
# results - список результатов анализа (см. api-format.md)

# Отключение
client.disconnect()
```

### 2. Собственный клиент (WebSocket)

```python
import json
import base64
import websocket

ws = websocket.WebSocket()
ws.connect("ws://localhost:8000/ws/rooms/test-room?name=test-user")

# Отправка кадра
_, buffer = cv2.imencode(".jpg", frame)
image_b64 = base64.b64encode(buffer).decode("utf-8")
ws.send(json.dumps({"image": image_b64}))

# Получение ответа
response = json.loads(ws.recv())
# response["image"] - base64 обработанного изображения
# response["results"] - результаты анализа
```

## HTTP API

### Проверка здоровья

```python
import requests

# Проверка здоровья
resp = requests.get("http://localhost:8000/health")
if resp.status_code == 200:
    print("Backend available")
```

### Получение списка комнат

```python
import requests

resp = requests.get("http://localhost:8000/rooms")
rooms = resp.json()  # ["room1", "room2", ...]
```

### Получение клиентов в комнате

```python
import requests

resp = requests.get("http://localhost:8000/rooms/my-room/clients")
clients = resp.json()
# [
#   {"name": "user1", "id_": "uuid-строка"},
#   {"name": "user2", "id_": "uuid-строка"}
# ]
```

### Создание комнаты

Комната создаётся автоматически при первом подключении клиента:

```python
# При подключении к /ws/rooms/{room_id}/stream
# комната создаётся автоматически
client.connect(room_id="new-room", name="user")
```

## Структура ответа для UI

```python
# results[0] - первое обнаруженное лицо
result = results[0]

emotion = result["emotion"]           # "Happy", "Neutral", "Sad", ...
confidence = result["confidence"]     # 0.0 - 1.0
bbox = result["bbox"]                 # (x1, y1, x2, y2)

# EAR (Eye Aspect Ratio)
ear = result.get("ear", {})
if ear:
    avg_ear = ear.get("avg_ear")      # 0.0 - 0.35
    attention = ear.get("attention_state")  # "Alert", "Normal", "Drowsy", "Very Drowsy"
    blink_count = ear.get("blink_count", 0)

# Head Pose
head_pose = result.get("head_pose", {})
if head_pose:
    pitch = head_pose.get("pitch")    # градусы
    yaw = head_pose.get("yaw")
    roll = head_pose.get("roll")
    attention = head_pose.get("attention_state")  # "Highly Attentive", ...

# Engagement
engagement = result.get("engagement", {})
if engagement:
    score = engagement.get("score")   # 0.0 - 1.0
    level = engagement.get("level")   # "High", "Medium", "Low", "Very Low"
    trend = engagement.get("trend")   # "rising", "falling", "stable"
```

## Обработка ошибок

```python
# Проверка доступности бэкенда
client = EngagementAPIClient()
if not client.check_health():
    print("Backend unavailable")

# Обработка ошибок при отправке
processed_frame, results = client.send_frame(frame)
if not results:  # пустой результат
    # Либо лицо не найдено, либо ошибка
    pass
```

## Обработка разрывов соединения

```python
# При разрыве соединения клиент выполняет:
# 1. Автоматический reconnect (1 попытка)
# 2. При неудаче возвращает (None, [])

processed_frame, results = client.send_frame(frame)
if processed_frame is None and not results:
    # Соединение разорвано, переподключение не удалось
    # Необходимо повторно вызвать client.connect()
    client.connect(room_id="my-room", name="user")
```

## Пример с веб-камерой

```python
import cv2

cap = cv2.VideoCapture(0)
client.connect(room_id="webcam-room", name="user")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed, results = client.send_frame(frame)
    
    if processed is not None:
        cv2.imshow("Processed", processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
client.disconnect()
```

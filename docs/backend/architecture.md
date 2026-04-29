# Архитектура системы

## Поток данных

```
Клиент ── WS ──► Сервер ──► Pipeline ──► WS ответ ──► Клиент
                                │
                                ▼
                          Redis Pub/Sub ──► Наблюдатель
```

### Этапы

| # | Описание                                                                     | Файл                |
|---|------------------------------------------------------------------------------|---------------------|
| 1 | Клиент отправляет Base64 image                                               | `stream.py:79`      |
| 2 | Сервер декодирует в OpenCV image                                             | `stream.py:91-103`  |
| 3 | Pipeline анализирует (FaceDetection → Emotion + EAR + HeadPose → Engagement) | `service.py:44-47`  |
| 4 | Ответ клиенту: `{image, results}`                                            | `stream.py:117-119` |
| 5 | Публикация в Redis Pub/Sub для наблюдателей                                  | `stream.py:112-115` |

---

## Обработка ошибок и граничные случаи

### Сводная таблица

| Сценарий                             | Источник              | Ответ API                                   | Действие клиента              |
|--------------------------------------|-----------------------|---------------------------------------------|-------------------------------|
| **Лицо не обнаружено**               | FaceDetection         | `{results: []}`                             | Показать "No face detected"   |
| **Redis недоступен (connect)**       | `add_client`          | WS 1011 + `{error: "Redis unavailable..."}` | Retry через 5 сек             |
| **Redis недоступен (send_frame)**    | `send_frame`          | Логирование, продолжение                    | Данные не в Pub/Sub           |
| **Invalid image**                    | Декодирование         | `{error: "Failed to decode image..."}`      | Повторить с корректным кадром |
| **Invalid JSON**                     | `receive_json`        | `{error: "Validation error..."}`            | Исправить формат              |
| **Connection broken**                | WebSocket disconnect  | `WebSocketDisconnect`                       | Reconnect                     |
| **Client not found (output_stream)** | `get_client`          | WS 1008 + `{error: "Client not found..."}`  | Проверить client_id           |
| **Room not found (clients)**         | `get_clients_in_room` | HTTP 404                                    | Комната не существует         |

### Детализация ответов

#### Лицо не обнаружено

```json
{
  "image": "base64_processed_image",
  "results": []
}
```

Кадр обрабатывается (накладываются debug-визуализации), но список результатов пуст. Клиент сам решает, как отобразить это состояние.

#### Redis недоступен при подключении

```json
{
  "error": "Redis unavailable, try again later"
}
```

Соединение закрывается с кодом `1011 (WS_1011_INTERNAL_ERROR)`.

```python
# stream.py:70-74
await websocket.send_json(ErrorResponse(error="Redis unavailable, try again later").model_dump())
await websocket.close(status.WS_1011_INTERNAL_ERROR)
```

#### Ошибка декодирования изображения

```json
{
  "error": "Failed to decode image: ..."
}
```

Кадр пропускается, цикл продолжает работать. Клиент может отправить следующий кадр.

---

## Управление комнатами (Room Management)

### Жизненный цикл комнаты

```
1. Создание          2. Активная фаза        3. Удаление
┌─────────────┐      ┌─────────────┐         ┌─────────────┐
│ connect()   │ ──►  │ клиент(ы)   │ ──────► │ remove_     │
│ add_client  │      │ отправляют  │   при   │ client()    │
│             │      │ кадры       │   всех  │ последнего  │
└─────────────┘      └─────────────┘ клиентах└─────────────┘
```

### Создание комнаты

Комната создаётся **автоматически** при первом подключении клиента:

```python
# stream.py:68-74
await room_service.add_client(client)

# rooms_and_clients.py:125-142
await redis.sadd("rooms", room_id)           # rooms: set всех комнат
await redis.sadd(f"room:{room_id}", ...)     # room:{id}: set клиентов
await redis.hset(f"client:{id}", ...)       # client:{id}: hash данных
```

### Жизненный цикл клиента

```
1. Подключение        2. Активен              3. Отключение
┌─────────────┐      ┌─────────────┐         ┌─────────────┐
│ /ws/rooms/  │ ──►  │ send_frame()│ ──────► │ WebSocket   │
│ {room_id}/  │      │ (Publish    │         │ disconnect  │
│ stream      │      │  to Pub/Sub)│         │ remove_client()
└─────────────┘      └─────────────┘         └─────────────┘
```

### Структура данных в Redis

| Key                         | Type    | Описание                |
|-----------------------------|---------|-------------------------|
| `rooms`                     | Set     | ID всех активных комнат |
| `room:{room_id}`            | Set     | ID клиентов в комнате   |
| `client:{client_id}`        | Hash    | `{name, source_closed}` |
| `client_stream:{client_id}` | Pub/Sub | Канал с кадрами клиента |

### Auto-cleanup

При удалении клиента (`remove_client`):

```python
# rooms_and_clients.py:173-204
await redis.delete(f"client:{client_id}")           # Удалить клиента
await redis.srem(f"room:{room_id}", client_id)      # Удалить из комнаты

# Если комната пуста — удалить
remaining = await redis.smembers(f"room:{room_id}")
if not remaining:
    await redis.srem("rooms", room_id)              # Удалить из списка комнат
    await redis.delete(f"room:{room_id}")            # Удалить саму комнату
```

### Output Stream (гипервизор)

Позволяет наблюдать за потоком другого клиента:

```
URL: ws://rooms/{room_id}/clients/{client_id}/output_stream

Поток данных:
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Client A    │      │   Redis     │      │  Observer   │
│ (источник)  │ ──►  │  Pub/Sub    │ ──►  │   (зритель) │
│             │      │             │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
```

Ответ содержит **исходный** и **обработанный** кадр:

```json
{
  "image_src": "base64_original",
  "image": "base64_processed",
  "results": [...]
}
```

### Реализация Pub/Sub

```python
# rooms_and_clients.py:264-280
async def send_frame(client, src_b64, prc_b64, results):
    payload = {"src": src_b64, "prc": prc_b64, "result": results}
    await redis.publish(f"client_stream:{client.id_}", json.dumps(payload))

# rooms_and_clients.py:282-309
async def get_frame_raw(client, timeout=0.0):
    pubsub = await redis.pubsub()
    await pubsub.subscribe(f"client_stream:{client.id_}")
    message = await pubsub.get_message(timeout=timeout)
    # ...
```

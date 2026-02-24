# Emotion Detection Backend

FastAPI-сервер для распознавания эмоций в реальном времени с использованием WebSocket.

## Возможности

- **Детекция лиц** через MediaPipe Face Detection
- **Распознавание эмоций** через EmotiEffLib (PyTorch)
- **Анализ EAR** (Eye Aspect Ratio) — детекция моргания и усталости
- **Оценка позы головы** — определение направления взгляда
- **WebSocket-стриминг** — обработка видеопотока в реальном времени
- **Управление комнатами** — изолированные сессии для нескольких клиентов

## Требования

- Python 3.12+
- CUDA (опционально, для ускорения PyTorch)

## Установка зависимостей

```bash
pip install .
```

## Запуск

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

### Сборка образа

```bash
docker build -t emotion-detection-backend .
```

### Запуск контейнера

```bash
docker run -d -p 8000:8000 --gpus all emotion-detection-backend
```

Для запуска без GPU:

```bash
docker run -d -p 8000:8000 emotion-detection-backend
```

### Docker Compose

Смотрите корневой `docker-compose.yaml` для запуска вместе с фронтендом.

## API Endpoints

### REST

| Method | Endpoint | Описание |
|--------|----------|----------|
| GET | `/health` | Проверка работоспособности |
| GET | `/rooms` | Список активных комнат |
| GET | `/rooms/{room_id}/clients` | Клиенты в комнате |

### WebSocket

| Endpoint | Описание |
|----------|----------|
| `/ws/rooms/{room_id}/stream` | Отправка кадров на анализ |
| `/ws/rooms/{room_id}/clients/{client_id}/output_stream` | Получение обработанного потока |

## Конфигурация

Настройки загружаются из переменных окружения или файла `.env`:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `app_version` | Версия приложения | 1.0.0 |
| `cors_allowed_origins` | CORS-источники | localhost:8501, localhost:63342 |
| `face_detection_min_confidence` | Порог детекции лица | 0.5 |
| `emotion_model_name` | Модель эмоций | enet_b2_8 |
| `emotion_device` | Устройство (cpu/cuda/auto) | auto |
| `ear_threshold` | Порог EAR | 0.25 |
| `head_pitch_attentive` | Порог наклона головы | 20.0 |

## Архитектура

```
app/
├── api/              # API маршруты
│   ├── stream.py     # WebSocket эндпоинты
│   └── room.py       # REST эндпоинты комнат
├── core/
│   └── config.py     # Конфигурация
├── schemas/          # Pydantic модели
└── services/
    ├── room.py              # Управление комнатами
    └── video_processing/    # Обработка видео
        ├── face_analysis_pipeline.py
        ├── face_detection.py
        ├── analyze_emotion.py
        ├── analyze_ear.py
        └── analyze_head_pose.py
```

## Тестирование

```bash
pytest
```

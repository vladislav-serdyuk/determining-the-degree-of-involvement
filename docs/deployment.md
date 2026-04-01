# Развёртывание бэкенда

## Docker Compose (рекомендуемый способ)

Запускает Redis и бэкенд одной командой:

```bash
docker-compose up -d
```

Остановка:
```bash
docker-compose down
```

---

## Ручная установка

### Требования

- **Python**: 3.10+
- **Redis**: 6.0+

### Установка зависимостей

```bash
cd backend
pip install .
```

### Запуск Redis

```bash
sudo systemctl start redis-server
# или
redis-server --requirepass password
```

### Запуск бэкенда

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Переменные окружения

| Переменная             | По умолчанию                                 | Описание                               |
|------------------------|----------------------------------------------|----------------------------------------|
| `REDIS_HOST`           | localhost                                    | Хост Redis                             |
| `REDIS_PORT`           | 6379                                         | Порт Redis                             |
| `REDIS_PASSWORD`       | password                                     | Пароль Redis                           |
| `CORS_ALLOWED_ORIGINS` | http://localhost:8501,http://localhost:63342 | Разрешённые CORS-источники             |
| `EMOTION_DEVICE`       | auto                                         | Устройство для PyTorch (cpu/cuda/auto) |

Подробнее в .env.example

---

## Проверка работоспособности

```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"1.0.0"}
```

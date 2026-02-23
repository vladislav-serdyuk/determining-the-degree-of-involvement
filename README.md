# determining-the-degree-of-involvement

Real-time emotion detection system with FastAPI backend and Streamlit frontend.

## Components

- **backend/**: FastAPI backend with WebSocket video streaming support
- **frontend/**: Streamlit frontend for video upload and webcam processing

## Quick Start

### Docker Compose

> You need to install nvidia-container-toolkit to support cuda

```bash
# Run backend
docker compose up -d --build 
```

### Manual

#### Install dependencies

```bash
cd backend && pip install -r requirements.txt
cd ../frontend && pip install -r requirements.txt
```

Required `Python 3.12+`:

```bash
# create venv with required version
python -3.12 -m venv venv
```

#### Run backend

> Start uvicorn only from `backend` directory. In another case its modules do not load properly

```bash
# if global uvicorn installed
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# from venv
cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run frontend (another terminal)

```bash
cd frontend && streamlit run emotion_detection_app.py
```

### Tests

To test server state just use `http://localhost:8000/health` in your browser and `http://localhost:8000/docs` to check
generated documentation.

To try API without streamlit app you need to open `../tests/test_ws_stream.html` page with browser
and click **Connect**, **Start Video** buttons.

> The server must be started before.

## Tech Stack

- MediaPipe (face detection)
- PyTorch+EmottiEffLib (emotion recognition)
- OpenCV (video processing)

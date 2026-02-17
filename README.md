# determining-the-degree-of-involvement

Real-time emotion detection system with FastAPI backend and Streamlit frontend.

## Components

- **server_app/**: FastAPI backend with WebSocket video streaming support
- **streamlit_app/**: Streamlit frontend for video upload and webcam processing

## Quick Start

### Docker Compose

> You need to install nvidia-container-toolkit to support cuda

```bash
# Run backend
cd server_app
docker compose up -d --build 
```

### Manual

#### Install dependencies

```bash
cd server_app && pip install -r requirements.txt
cd ../streamlit_app && pip install -r requirements.txt
```

Required `Python 3.12+`:

```bash
# create venv with required version
python -3.12 -m venv venv
```

#### Run backend

> Start uvicorn only from `server_app` directory. In another case its modules do not load properly

```bash
# if global uvicorn installed
cd server_app && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# from venv
cd server_app && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run frontend (another terminal)

```bash
cd streamlit_app && streamlit run emotion_detection_app.py
```

### Tests

To test server state just use `http://localhost:8000/health` in your browser and `http://localhost:8000/docs` to check
generated documentation.

To try API without streamlit app you need to open `../tests/test_ws_stream.html` page with browser and click **Connect**, **Start Video** buttons.

> The server must be started before.

## Tech Stack

- MediaPipe (face detection)
- PyTorch+EmottiEffLib (emotion recognition)
- OpenCV (video processing)

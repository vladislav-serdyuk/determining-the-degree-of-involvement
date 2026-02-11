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

```bash
# Install dependencies
cd server_app && pip install -r requirements.txt
cd ../streamlit_app && pip install -r requirements.txt

# Run backend
cd ../server_app && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run frontend (in another terminal)
cd streamlit_app && streamlit run emotion_detection_app.py
```

## Tech Stack

- MediaPipe (face detection)
- PyTorch (emotion recognition)
- OpenCV (video processing)

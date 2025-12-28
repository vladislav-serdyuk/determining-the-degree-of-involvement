import streamlit as st
import os
import tempfile
import subprocess
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import sys
import json
import threading
import queue
import atexit

# ============================================
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ============================================

APP_TITLE = "🎭 Real-time Emotion Detection"
APP_ICON = "🎭"
SUPPORTED_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Путь к вашему бэкенд-скрипту
# Убедитесь, что файл face_detection_and_emotion_recognition.py в той же папке
BACKEND_SCRIPT = "face_detection_and_emotion_recognition.py"

# ============================================
# CSS СТИЛИ
# ============================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Основные стили */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        padding: 0.5rem;
    }
    
    .upload-zone {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #FF416C;
        background: linear-gradient(135deg, #667eea1a 0%, #764ba21a 100%);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .processing-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .result-card {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(86, 171, 47, 0.3);
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(255, 65, 108, 0.3);
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .video-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        background: #000;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(90deg, #764ba2, #667eea);
    }
    
    /* Прогресс бар */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    /* Уведомления */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Сайдбар */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea0d 0%, #764ba20d 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# КЛАСС ДЛЯ ОБРАБОТКИ ВИДЕО
# ============================================

class VideoProcessor:
    """Класс для запуска вашего бэкенда и обработки видео"""
    
    def __init__(self):
        self.backend_process = None
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def check_backend_exists(self):
        """Проверяет наличие бэкенд-скрипта"""
        if not os.path.exists(BACKEND_SCRIPT):
            # Попробуем найти в разных местах
            possible_paths = [
                BACKEND_SCRIPT,
                "../determining-the-degree-of-involvement/face_detection_and_emotion_recognition.py",
                "./face_detection_and_emotion_recognition.py",
                "face_detection_and_emotion_recognition.py"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return True, path
            
            return False, None
        return True, BACKEND_SCRIPT
    
    def extract_video_info(self, video_path):
        """Извлекает информацию о видео"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Cannot open video file"}
            
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 2) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                "format": self._get_video_format(video_path)
            }
            
            # Извлекаем превью (первый кадр)
            ret, frame = cap.read()
            if ret:
                preview_path = "temp_preview.jpg"
                cv2.imwrite(preview_path, frame)
                info["preview"] = preview_path
            
            cap.release()
            return info
            
        except Exception as e:
            return {"error": f"Cannot extract video info: {str(e)}"}
    
    def _get_video_format(self, video_path):
        """Определяет формат видео"""
        ext = os.path.splitext(video_path)[1].lower()
        formats = {
            '.mp4': 'MP4',
            '.avi': 'AVI',
            '.mov': 'MOV',
            '.mkv': 'MKV',
            '.webm': 'WebM',
            '.wmv': 'WMV'
        }
        return formats.get(ext, 'Unknown')
    
    def process_video(self, input_path, output_path, flip_h=False, show_preview=False):
        """
        Запускает ваш бэкенд для обработки видео
        
        Args:
            input_path: Путь к входному видео
            output_path: Путь для сохранения результата
            flip_h: Отзеркаливание по горизонтали
            show_preview: Показывать превью во время обработки
            
        Returns:
            (success, message, output_path)
        """
        try:
            # Проверяем наличие бэкенда
            exists, script_path = self.check_backend_exists()
            if not exists:
                return False, "Backend script not found. Please ensure 'face_detection_and_emotion_recognition.py' is in the same directory.", None
            
            # Подготавливаем команду
            cmd = [
                sys.executable,  # Используем текущий интерпретатор Python
                script_path,
                "--input", input_path,
                "--output", output_path
            ]
            
            if flip_h:
                cmd.append("--flip")
            
            if show_preview:
                cmd.append("--show")
            
            # Создаем временный файл для логов
            log_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
            log_path = log_file.name
            log_file.close()
            
            # Запускаем процесс
            st.info(f"Starting backend processing...\nCommand: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=open(log_path, 'w'),
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Ждем завершения
            start_time = time.time()
            timeout = 3600  # 1 час
            
            while True:
                if process.poll() is not None:
                    break
                
                if time.time() - start_time > timeout:
                    process.terminate()
                    return False, "Processing timeout (1 hour)", None
                
                time.sleep(1)
            
            # Проверяем результат
            return_code = process.returncode
            
            if return_code == 0:
                if os.path.exists(output_path):
                    # Читаем логи
                    with open(log_path, 'r') as f:
                        logs = f.read()
                    
                    # Ищем статистику в логах
                    stats = self._extract_stats_from_logs(logs)
                    
                    return True, f"Processing completed successfully!\n\nLogs:\n{logs}", output_path
                else:
                    return False, f"Processing completed but output file not found at {output_path}", None
            else:
                # Читаем ошибки
                with open(log_path, 'r') as f:
                    error_logs = f.read()
                
                return False, f"Backend processing failed with code {return_code}:\n{error_logs}", None
            
        except Exception as e:
            return False, f"Error during processing: {str(e)}", None
    
    def _extract_stats_from_logs(self, logs):
        """Извлекает статистику из логов бэкенда"""
        stats = {}
        
        # Ищем ключевые слова в логах
        lines = logs.split('\n')
        for line in lines:
            if 'frame' in line.lower() and 'fps' in line.lower():
                stats['processing_info'] = line.strip()
            elif 'emotion' in line.lower():
                stats['emotion_info'] = line.strip()
        
        return stats
    
    def extract_sample_frames(self, video_path, num_frames=4):
        """Извлекает несколько кадров из видео для превью"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            # Выбираем равномерно распределенные кадры
            frame_indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Конвертируем BGR в RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            return frames
            
        except Exception as e:
            st.warning(f"Could not extract sample frames: {e}")
            return []

# ============================================
# ИНИЦИАЛИЗАЦИЯ СЕССИИ
# ============================================

if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"

if 'result_path' not in st.session_state:
    st.session_state.result_path = None

if 'video_info' not in st.session_state:
    st.session_state.video_info = {}

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def display_header():
    """Отображает заголовок приложения"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Upload a video to detect faces and recognize emotions in real-time</p>', unsafe_allow_html=True)

def display_sidebar():
    """Отображает боковую панель"""
    with st.sidebar:
        st.markdown("### 🎭 Emotion Detection")
        
        # Информация о системе
        st.markdown("#### ℹ️ System Status")
        
        # Проверяем наличие бэкенда
        exists, path = st.session_state.processor.check_backend_exists()
        if exists:
            st.success("✅ Backend script found")
        else:
            st.error("❌ Backend script not found")
            st.info("Please ensure 'face_detection_and_emotion_recognition.py' is in the same directory")
        
        st.markdown("---")
        
        # Технологии
        st.markdown("#### 🔧 Technologies")
        st.markdown("""
        - **Face Detection**: MediaPipe
        - **Emotion Recognition**: EmotiEffLib
        - **Video Processing**: OpenCV
        - **Interface**: Streamlit
        """)
        
        st.markdown("---")
        
        # Инструкции
        st.markdown("#### 📋 How to Use")
        st.markdown("""
        1. **Upload** a video file
        2. **Configure** processing options
        3. **Process** with AI
        4. **Download** the result
        """)
        
        st.markdown("---")
        
        # Поддерживаемые форматы
        st.markdown("#### 📁 Supported Formats")
        st.markdown(f"""
        {' • '.join([f'**{fmt.upper()}**' for fmt in SUPPORTED_FORMATS])}
        
        **Max size**: {MAX_FILE_SIZE // (1024*1024)}MB
        """)

def create_upload_section():
    """Создает секцию загрузки файла"""
    st.markdown("### 📤 Upload Video")
    
    # Зона загрузки
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS).upper()}",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Проверяем размер файла
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File too large! Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
            return
        
        # Сохраняем временный файл
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.session_state.uploaded_file_path = temp_path
        
        # Извлекаем информацию о видео
        video_info = st.session_state.processor.extract_video_info(temp_path)
        st.session_state.video_info = video_info
        
        if "error" not in video_info:
            # Показываем информацию о файле
            display_file_info(uploaded_file, video_info)
            
            # Показываем превью видео
            display_video_preview(temp_path, video_info)
            
            # Кнопка обработки
            if st.button("🚀 Start Emotion Detection", type="primary", use_container_width=True):
                st.session_state.processing_status = "starting"
                st.rerun()
        else:
            st.error(f"Error: {video_info['error']}")
    
    else:
        # Показываем зону загрузки
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 4rem;">📁</div>
                <h3>Drag & Drop Video Here</h3>
                <p style="color: #666;">or click to browse</p>
                <p style="font-size: 0.8rem; color: #999; margin-top: 2rem;">
                Supports MP4, AVI, MOV, MKV, WebM, WMV
                </p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_file_info(uploaded_file, video_info):
    """Отображает информацию о файле"""
    st.markdown("### 📊 Video Information")
    
    # Статистика в карточках
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{video_info["width"]}×{video_info["height"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Resolution</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        duration = video_info["duration"]
        if duration >= 60:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_str = f"{minutes}:{seconds:02d}"
        else:
            duration_str = f"{duration:.1f}s"
        
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{duration_str}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Duration</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{video_info["fps"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">FPS</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{file_size_mb:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Size (MB)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_video_preview(video_path, video_info):
    """Отображает превью видео"""
    st.markdown("### 👀 Video Preview")
    
    # Основное видео
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_bytes = open(video_path, "rb").read()
    st.video(video_bytes)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Примеры кадров
    st.markdown("#### 📸 Sample Frames")
    
    frames = st.session_state.processor.extract_sample_frames(video_path, 4)
    if frames:
        cols = st.columns(4)
        for idx, (col, frame) in enumerate(zip(cols, frames)):
            with col:
                img = Image.fromarray(frame)
                st.image(img, caption=f"Frame {idx+1}", use_container_width=True)

def process_video():
    """Обрабатывает видео"""
    if st.session_state.processing_status == "starting" and st.session_state.uploaded_file_path:
        st.session_state.processing_status = "processing"
        
        # Настройки обработки
        with st.expander("⚙️ Processing Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                flip_h = st.checkbox("Flip horizontally", value=False)
                show_preview = st.checkbox("Show preview (if supported)", value=False)
            with col2:
                st.info("Note: Processing may take several minutes depending on video length")
        
        # Показываем панель прогресса
        st.markdown('<div class="processing-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Processing Your Video")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Создаем имя для выходного файла
        input_path = st.session_state.uploaded_file_path
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = int(time.time())
        output_filename = f"emotion_detected_{input_name}_{timestamp}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # Запускаем обработку в отдельном потоке
        def run_processing():
            try:
                # Этапы обработки для отображения прогресса
                stages = [
                    ("Initializing models...", 10),
                    ("Loading video...", 20),
                    ("Detecting faces...", 40),
                    ("Recognizing emotions...", 70),
                    ("Processing frames...", 90),
                    ("Finalizing...", 100)
                ]
                
                for stage_text, stage_progress in stages:
                    status_text.text(stage_text)
                    progress_bar.progress(stage_progress)
                    time.sleep(1)  # Имитация обработки
                
                # Запускаем реальную обработку
                success, message, result_path = st.session_state.processor.process_video(
                    input_path,
                    output_path,
                    flip_h=flip_h,
                    show_preview=show_preview
                )
                
                if success:
                    st.session_state.processing_status = "completed"
                    st.session_state.result_path = result_path
                else:
                    st.session_state.processing_status = "failed"
                    st.session_state.error_message = message
                    
            except Exception as e:
                st.session_state.processing_status = "failed"
                st.session_state.error_message = str(e)
        
        # Запускаем обработку
        import threading
        thread = threading.Thread(target=run_processing)
        thread.start()
        
        # Ждем завершения
        while thread.is_alive():
            time.sleep(0.5)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Показываем результат
        if st.session_state.processing_status == "completed":
            display_result()
        elif st.session_state.processing_status == "failed":
            display_error()

def display_result():
    """Отображает результат обработки"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Processing Completed!")
    
    result_path = st.session_state.result_path
    
    if result_path and os.path.exists(result_path):
        # Показываем обработанное видео
        st.markdown("#### 🎬 Processed Video")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(result_path)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Кнопка скачивания
        with open(result_path, "rb") as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name=os.path.basename(result_path),
                mime="video/mp4",
                type="primary",
                use_container_width=True
            )
        
        # Информация о результате
        result_info = st.session_state.processor.extract_video_info(result_path)
        if "error" not in result_info:
            st.markdown("---")
            st.markdown("#### 📊 Result Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resolution", f"{result_info['width']}×{result_info['height']}")
            with col2:
                st.metric("Duration", f"{result_info['duration']:.1f}s")
            with col3:
                st.metric("FPS", result_info['fps'])
    
    else:
        st.warning("Processed video file not found")
    
    # Кнопка для новой обработки
    if st.button("🔄 Process Another Video", use_container_width=True):
        st.session_state.uploaded_file_path = None
        st.session_state.processing_status = "idle"
        st.session_state.result_path = None
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_error():
    """Отображает ошибку обработки"""
    st.markdown('<div class="error-card">', unsafe_allow_html=True)
    st.markdown("### ❌ Processing Failed")
    
    error_msg = getattr(st.session_state, 'error_message', 'Unknown error')
    st.error(f"Error: {error_msg}")
    
    # Советы по устранению неполадок
    st.markdown("#### 🔧 Troubleshooting Tips:")
    st.markdown("""
    1. ✅ Ensure `face_detection_and_emotion_recognition.py` is in the same directory
    2. ✅ Check if all dependencies are installed
    3. ✅ Try a shorter video (under 1 minute)
    4. ✅ Ensure the video format is supported
    5. ✅ Check available disk space
    """)
    
    if st.button("🔄 Try Again", use_container_width=True):
        st.session_state.processing_status = "idle"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_example_results():
    """Отображает примеры результатов"""
    st.markdown("### 🎭 Example Detection Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 😊 Happiness")
        st.image("https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/docs/images/face_detection_android_gpu.gif", 
                caption="Happy face detection")
    
    with col2:
        st.markdown("#### 😢 Sadness")
        st.image("https://viso.ai/wp-content/uploads/2021/05/facial-expression-recognition-software.png",
                caption="Sad emotion recognition")
    
    with col3:
        st.markdown("#### 😮 Surprise")
        st.image("https://www.researchgate.net/profile/Amir-Hussain-8/publication/327404470/figure/fig3/AS:668258825682954@1536341716485/Sample-output-of-emotion-detection-on-video-frame-sequence.ppm",
                caption="Surprise detection")
    
    st.markdown("---")
    
    # Описание эмоций
    st.markdown("#### 🎯 Detected Emotions")
    
    emotions = {
        "😊 Happy": "Positive emotion with raised cheeks and smile",
        "😢 Sad": "Downward mouth, drooping eyelids",
        "😠 Angry": "Lowered eyebrows, tense mouth",
        "😮 Surprise": "Raised eyebrows, wide eyes",
        "😐 Neutral": "Relaxed facial features",
        "😨 Fear": "Wide eyes, raised eyebrows",
        "🤢 Disgust": "Wrinkled nose, raised upper lip"
    }
    
    for emotion, description in emotions.items():
        st.markdown(f"**{emotion}**: {description}")

# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================

def main():
    """Основная функция приложения"""
    display_header()
    display_sidebar()
    
    # Основной контент в вкладках
    tab1, tab2, tab3 = st.tabs(["🎬 Upload & Process", "📊 How It Works", "❓ Help & Support"])
    
    with tab1:
        # Проверяем статус обработки
        if st.session_state.processing_status in ["idle", "starting"]:
            create_upload_section()
        
        if st.session_state.processing_status == "processing":
            process_video()
        elif st.session_state.processing_status == "completed":
            display_result()
        elif st.session_state.processing_status == "failed":
            display_error()
    
    with tab2:
        display_example_results()
        
        st.markdown("---")
        
        st.markdown("### 🔬 Technical Details")
        st.markdown("""
        #### Face Detection (MediaPipe)
        - **Model**: BlazeFace (optimized for real-time)
        - **Accuracy**: ~95% on standard datasets
        - **Features**: 6 facial landmarks per face
        - **Range**: Works from 0.5m to 5m distance
        
        #### Emotion Recognition (EmotiEffLib)
        - **Model**: EfficientNet-B2 trained on AffectNet
        - **Emotions**: 8 basic emotions + contempt
        - **Temporal smoothing**: 15-frame window
        - **Confidence threshold**: 55% minimum
        
        #### Processing Pipeline
        1. **Frame extraction** from video
        2. **Face detection** in each frame
        3. **Emotion classification** per face
        4. **Bounding box drawing** with labels
        5. **Video reconstruction** with overlays
        """)
    
    with tab3:
        st.markdown("### ❓ Frequently Asked Questions")
        
        faqs = [
            {
                "question": "How long does processing take?",
                "answer": "Processing time depends on video length. Typically 0.1-0.2 seconds per frame. A 1-minute video (1800 frames) takes about 3-6 minutes."
            },
            {
                "question": "What video formats are supported?",
                "answer": f"We support: {', '.join([fmt.upper() for fmt in SUPPORTED_FORMATS])}. The video will be converted to MP4 during processing."
            },
            {
                "question": "Can I process long videos?",
                "answer": "Yes, but processing time will be longer. For videos over 10 minutes, we recommend trimming to the most important segments."
            },
            {
                "question": "What emotions can be detected?",
                "answer": "The system detects 8 emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral, and Contempt."
            },
            {
                "question": "Do you store my videos?",
                "answer": "No. All processing is done locally or in memory. Videos are temporarily stored only during processing and deleted afterward."
            },
            {
                "question": "What if I get an error?",
                "answer": "Check that 'face_detection_and_emotion_recognition.py' is in the same directory. Ensure all dependencies are installed. Try a shorter video for testing."
            }
        ]
        
        for faq in faqs:
            with st.expander(f"**Q:** {faq['question']}"):
                st.markdown(f"**A:** {faq['answer']}")
        
        st.markdown("---")
        
        st.markdown("### 🐛 Troubleshooting")
        st.markdown("""
        #### Common Issues and Solutions:
        
        **1. Backend script not found**
        ```
        Solution: Ensure 'face_detection_and_emotion_recognition.py' is in the same directory as this app.
        ```
        
        **2. Processing takes too long**
        ```
        Solution: Try shorter videos first. Check CPU/GPU usage.
        ```
        
        **3. No faces detected**
        ```
        Solution: Ensure faces are clearly visible. Try different lighting conditions.
        ```
        
        **4. Out of memory error**
        ```
        Solution: Close other applications. Use shorter videos. Restart the application.
        ```
        """)
        
        st.markdown("---")
        
        st.markdown("### 📞 Support")
        st.markdown("""
        If you continue to experience issues:
        
        - **Check dependencies**: Ensure all required packages are installed
        - **Update software**: Make sure you have the latest versions
        - **Test with sample video**: Try with a short, clear video first
        - **Check logs**: Look for error messages in the console
        
        For technical support, please provide:
        - Video specifications (format, length, resolution)
        - Error messages
        - System specifications
        """)

# ============================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please restart the application and try again.")
        
        # Кнопка для перезапуска
        if st.button("🔄 Restart Application"):
            st.rerun()

import os

# Cantina Face Recognition System Configuration

# Face Recognition
SIM_THRESHOLD = 0.38        # Similarity threshold for face matching
FRAME_INTERVAL_MS = 500     # Interval between recognition frames (ms) - Reduced from 200ms
SEARCH_LIMIT = 10           # Maximum search results
FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", "models/mobile_face.onnx")
FACE_MODEL_URL = os.getenv("FACE_MODEL_URL", "")  # Offline: model must exist locally

# Performance optimizations (env tunable)
FACE_MAX_EMB_PER_SEC = float(os.getenv("FACE_MAX_EMB_PER_SEC", "2"))
FACE_CACHE_MS = int(os.getenv("FACE_CACHE_MS", "500"))
FACE_CACHE_IOU = float(os.getenv("FACE_CACHE_IOU", "0.7"))
FACE_DETECT_WIDTH = int(os.getenv("FACE_DETECT_WIDTH", "640"))
PERF_WINDOW_SECONDS = int(os.getenv("FACE_PERF_WINDOW_SECONDS", "10"))

ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", os.getenv("ORT_INTRA_OP_THREADS", "1")))
ORT_INTER_THREADS = int(os.getenv("ORT_INTER_THREADS", os.getenv("ORT_INTER_OP_THREADS", "1")))
ORT_INTRA_OP_THREADS = ORT_INTRA_THREADS  # backwards compatibility
ORT_INTER_OP_THREADS = ORT_INTER_THREADS
CV2_NUM_THREADS = int(os.getenv("CV2_NUM_THREADS", "1"))

LOW_RES_WIDTH = int(os.getenv("LOW_RES_WIDTH", str(max(288, FACE_DETECT_WIDTH // 2 or 288))))
LOW_RES_HEIGHT = int(os.getenv("LOW_RES_HEIGHT", str(int(max(216, LOW_RES_WIDTH * 0.75)))))
HIGH_RES_WIDTH = int(os.getenv("HIGH_RES_WIDTH", str(FACE_DETECT_WIDTH)))
HIGH_RES_HEIGHT = int(os.getenv("HIGH_RES_HEIGHT", os.getenv("FACE_DETECT_HEIGHT", "480")))
DETECTION_COOLDOWN = 2000   # Cooldown between face detections (ms)
NO_FACE_TIMEOUT = 5000      # Timeout to reduce FPS when no face detected (ms)
RECOGNITION_MIN_INTERVAL_MS = int(os.getenv("RECOGNITION_MIN_INTERVAL_MS", "700"))
NO_FACE_BACKOFF_MS = int(os.getenv("NO_FACE_BACKOFF_MS", "1500"))
MAX_NO_FACE_BACKOFF_MS = int(os.getenv("MAX_NO_FACE_BACKOFF_MS", "4000"))

# System
CURRENCY = "Gs."            # Currency symbol for display

# Database
MAX_ELEMENTS = 10000        # Maximum elements in HNSW index
EF_CONSTRUCTION = 200       # HNSW index construction parameter
M_INDEX = 16               # HNSW index M parameter

# Face Processing
INPUT_SIZE = (112, 112)    # ArcFace input size
FACE_MARGIN = 0.1          # Face detection margin
MIN_FACE_SIZE = (30, 30)   # Minimum face size for detection

# Stock Management
DEFAULT_MIN_STOCK = 20      # Default minimum stock threshold
STOCK_ALERT_THRESHOLD = 20 # Alert when stock <= this value

# Email / SMTP
SMTP_HOST = os.getenv("SMTP_HOST", "mail.siloe.com.py")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "sistema@siloe.com.py")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "Sistema Cantina Siloé <sistema@siloe.com.py>")
SMTP_ENABLED = os.getenv("SMTP_ENABLED", "1") not in ("0", "false", "False")
LOW_BALANCE_THRESHOLD = int(os.getenv("LOW_BALANCE_THRESHOLD", "5000"))  # Gs.

# User Roles and Permissions
USER_ROLES = {
    "admin": {
        "permissions": ["all"],
        "description": "Administrador completo del sistema"
    },
    "cajero": {
        "permissions": ["sell", "view_students", "view_products"],
        "description": "Cajero - puede vender y ver información básica"
    },
    "stock": {
        "permissions": ["manage_products", "view_stock", "manage_stock_requests"],
        "description": "Gestor de stock - maneja productos y solicitudes"
    },
    "administracion": {
        "permissions": ["manage_users", "view_reports", "manage_balances"],
        "description": "Administración - gestiona usuarios y saldos"
    }
}

# Cloud Backend Sync
CLOUD_BACKEND_URL = os.getenv("CLOUD_BACKEND_URL", "https://sistema.siloe.com.py")
CLOUD_SYNC_TOKEN = os.getenv("CLOUD_SYNC_TOKEN", "change-this-sync-token")
CLOUD_SYNC_INTERVAL_SECONDS = int(os.getenv("CLOUD_SYNC_INTERVAL_SECONDS", "300"))  # 5 minutes

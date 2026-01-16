# Cantina Face Recognition System Configuration

# Face Recognition
SIM_THRESHOLD = 0.38        # Similarity threshold for face matching
FRAME_INTERVAL_MS = 500     # Interval between recognition frames (ms) - Reduced from 200ms
SEARCH_LIMIT = 10           # Maximum search results

# Performance optimizations
LOW_RES_WIDTH = 320         # Low resolution width for detection
LOW_RES_HEIGHT = 240        # Low resolution height for detection
HIGH_RES_WIDTH = 640        # High resolution width for recognition
HIGH_RES_HEIGHT = 480       # High resolution height for recognition
DETECTION_COOLDOWN = 2000   # Cooldown between face detections (ms)
NO_FACE_TIMEOUT = 5000      # Timeout to reduce FPS when no face detected (ms)

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

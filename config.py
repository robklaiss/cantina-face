# Cantina Face Recognition System Configuration

# Face Recognition
SIM_THRESHOLD = 0.38        # Similarity threshold for face matching
FRAME_INTERVAL_MS = 200     # Interval between recognition frames (ms)
SEARCH_LIMIT = 10           # Maximum search results

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

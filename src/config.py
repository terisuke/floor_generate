"""
Configuration file for floor plan generation project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Data subdirectories
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted"
NORMALIZED_DIR = DATA_DIR / "normalized"
TRAINING_DIR = DATA_DIR / "training"
VALIDATION_DIR = DATA_DIR / "validation"

# Output subdirectories
GENERATED_DIR = OUTPUT_DIR / "generated"
SVG_DIR = OUTPUT_DIR / "svg"
DXF_DIR = OUTPUT_DIR / "dxf"
FREECAD_DIR = OUTPUT_DIR / "freecad"

# Model settings
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR))

# Floor plan constraints
MAX_ROOMS = int(os.getenv("MAX_ROOMS", "20"))
MIN_ROOM_SIZE = float(os.getenv("MIN_ROOM_SIZE", "10"))  # in square meters
MAX_ROOM_SIZE = float(os.getenv("MAX_ROOM_SIZE", "100"))  # in square meters

# Default floor plan dimensions (in meters)
DEFAULT_FLOOR_WIDTH = 30
DEFAULT_FLOOR_HEIGHT = 20

# Room types
ROOM_TYPES = [
    "office",
    "meeting_room",
    "kitchen",
    "bathroom",
    "storage",
    "reception",
    "lounge",
    "server_room"
]

# Room size constraints by type (min, max in square meters)
ROOM_SIZE_CONSTRAINTS = {
    "office": (10, 50),
    "meeting_room": (15, 40),
    "kitchen": (8, 25),
    "bathroom": (3, 10),
    "storage": (2, 15),
    "reception": (15, 60),
    "lounge": (20, 80),
    "server_room": (5, 20)
}

# Ensure directories exist
for directory in [RAW_PDF_DIR, EXTRACTED_DIR, NORMALIZED_DIR, TRAINING_DIR, 
                  VALIDATION_DIR, GENERATED_DIR, SVG_DIR, DXF_DIR, FREECAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

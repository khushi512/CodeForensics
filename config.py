# CodeForensics Configuration
import os
import logging
from pathlib import Path
from datetime import datetime

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, REPOS_DIR, DATASETS_DIR, MODELS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('codeforensics')

# Repository configuration
REPOS = [
    "pallets/flask",
    "pallets/click",
    "requests/requests",
    "psf/black",
    "encode/httpx"
]

# Processing limits (laptop-friendly)
CLONE_DEPTH = 500
MAX_COMMITS = 400
BATCH_SIZE = 100

# Bug-fix detection keywords
BUG_KEYWORDS = ['fix', 'bug', 'issue', 'patch', 'error', 'crash', 'broken', 'resolve', 'closes', 'fixes']

# Feature engineering settings
MAX_FILE_CHURN_LOOKBACK = 50
COMPLEXITY_LANGUAGES = ['.py']  # Extend later: ['.py', '.js', '.ts']
MIN_COMMIT_SIZE = 1
MAX_COMMIT_SIZE = 1000

# Database (SQLite local, PostgreSQL on Render)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/codeforensics.db")

# Feature columns for model
FEATURE_COLUMNS = [
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'files_changed',
    'py_files_modified',
    'lines_added',
    'lines_deleted',
    'net_lines',
    'complexity_delta',
    'avg_file_churn'
]

# Model hyperparameters
MODEL_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 31,
    'min_child_samples': 20,
    'class_weight': 'balanced',
    'verbose': -1
}

# Evaluation settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIGH_RISK_THRESHOLD = 0.7  # For VSCode extension warnings
MEDIUM_RISK_THRESHOLD = 0.4

# Dataset versioning
def get_dataset_filename(suffix: str = "") -> str:
    """Generate timestamped dataset filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"dataset_{timestamp}"
    if suffix:
        base += f"_{suffix}"
    return f"{base}.csv"

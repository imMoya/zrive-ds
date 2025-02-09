"Config (inspo from https://mlops-coding-course.fmind.dev/2.%20Prototyping/2.2.%20Configs.html)"

from pathlib import Path

# Define paths for caching and training data
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(ROOT, 'datasets/module_3/')
DATA_PATH = Path(DATA, 'feature_frame_filtered.csv')
MODELS = Path(ROOT, 'models/')
SCALER_PATH = Path(MODELS, 'scaler.pkl')

# Configure random state for reproducibility
RANDOM_STATE = 42

# Define dataset columns for feature engineering
DATETIME_COL = 'created_at'
USER_COL = 'user_id'
ORDER_COL = 'user_order_seq'
TARGET_COL = 'outcome'
CATEGORICAL_COLS = ['product_type']
DROP_COLS = ['created_at', 'order_date', 'vendor']

# Setup dataset parameters for testing and shuffling
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Parameters for pipeline configuration
SCALER = 'standard'

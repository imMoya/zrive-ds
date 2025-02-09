"Config (inspo from https://mlops-coding-course.fmind.dev/2.%20Prototyping/2.2.%20Configs.html)"

from pathlib import Path

# Define paths for caching and training data
DATA = Path('datasets/module_3/')
MODELS = Path('models/')
SCALER_PATH = Path(MODELS, 'scaler.pkl')

# Configure random state for reproducibility
RANDOM = 42

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
SCALER = "standard"

PREPROCESSOR_CONFIG = {
    'datetime_col': 'created_at',
    'user_col': 'user_id',
    'order_col': 'user_order_seq',
    'target_col': 'outcome',
    'categorical_cols': ['product_type'],
    'drop_cols': ['created_at', 'order_date', 'vendor'],
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'scaler_path': 'models/scaler.pkl',
}

PIPELINE_CONFIG = {'data_path': 'datasets/module_3/feature_frame_filtered.csv'}

"Config (inspo from https://mlops-coding-course.fmind.dev/2.%20Prototyping/2.2.%20Configs.html)"

from pathlib import Path

# Define paths for caching and training data
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(ROOT, 'datasets/module_3/')
DATA_PATH = Path(DATA, 'feature_frame.csv')
FIG_PATH = Path(ROOT, 'figures/module_3/')
MODELS = Path(ROOT, 'models/')
SCALER_PATH = Path(MODELS, 'scaler.pkl')

# Configure random state for reproducibility
RANDOM_STATE = 42

# Define dataset columns for feature engineering
DATETIME_COL = 'created_at'
DATETIME_DAY = 'order_date'
USER_COL = 'user_id'
ORDER_COL = 'order_id'
TARGET_COL = 'outcome'
CATEGORICAL_COLS = ['product_type']
DROP_COLS = ['created_at', 'order_date', 'vendor']

# Setup dataset parameters for testing and shuffling
TRAIN_RATIO = 0.85
VAL_RATIO = 0.0
TEST_RATIO = 0.15

# Parameters for pipeline configuration
SCALER = 'standard'

# Define model parameters for training
MODEL_TYPE = 'logistic_regression'
HYPERPARAMS = {
    'C': 0.0001
}

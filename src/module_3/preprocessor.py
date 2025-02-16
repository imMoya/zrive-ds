import logging

import joblib
import numpy as np
import pandas as pd
from config import (
    CATEGORICAL_COLS,
    DATETIME_COL,
    DROP_COLS,
    SCALER,
    SCALER_PATH,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    ORDER_COL,
    VAL_RATIO,
)
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Configure Logger
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCALERS = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler(),
}

VAL_RATIO_EPSILON = 0.001 # Minimum ratio for validation set


class PreprocessorConfig(BaseModel):
    datetime_col: str = DATETIME_COL
    order_col: str = ORDER_COL
    target_col: str = TARGET_COL
    categorical_cols: list[str] = CATEGORICAL_COLS
    drop_cols: list[str] = DROP_COLS
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    scaler: str = SCALER
    scaler_path: str = SCALER_PATH


class Preprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config

    def extract_time_features(
        self, df: pd.DataFrame, datetime_col: str
    ) -> pd.DataFrame:
        """Extracts time-related features from a datetime column."""
        if datetime_col not in df.columns:
            logger.error(f"Column '{datetime_col}' not found in DataFrame.")
            raise ValueError(f"Column '{datetime_col}' not found.")

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df['year'] = df[datetime_col].dt.year
        df['month'] = df[datetime_col].dt.month
        df['day'] = df[datetime_col].dt.day
        df['hour'] = df[datetime_col].dt.hour
        df['minute'] = df[datetime_col].dt.minute
        logger.info(f"Extracted time features from '{datetime_col}'.")
        return df
    
    def get_max_train_date(self, df: pd.DataFrame) -> int:
        """Calculates the idx of the last user in the training set, ordered by date."""
        unique_orders = df.groupby(self.config.datetime_col)[self.config.order_col].nunique()
        return unique_orders[unique_orders.cumsum() <= self.config.train_ratio*len(unique_orders)].idxmax()
    
    def get_max_val_date(self, df: pd.DataFrame) -> int:
        """Calculates the idx of the last user in the validation set, ordered by date."""
        unique_orders = df.groupby(self.config.datetime_col)[self.config.order_col].nunique()
        return unique_orders[unique_orders.cumsum() > (self.config.train_ratio+self.config.val_ratio)*len(unique_orders)].idxmin()
    
    @staticmethod
    def get_X_y(df: pd.DataFrame, X_cols: list[str], y_col: str) -> tuple[np.ndarray, np.ndarray]:
        """Extracts features and target from the DataFrame."""
        return df[X_cols], df[y_col].values

    @staticmethod
    def save_scaler(scaler, scaler_filename: str):
        """Saves the fitted scaler to a file."""
        try:
            joblib.dump(scaler, scaler_filename)
            logger.info(f'Scaler saved to {scaler_filename}')
        except Exception as e:
            logger.error(f'Error saving scaler: {e}')
            raise

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray | None, np.ndarray | None],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Applies full preprocessing pipeline to the DataFrame."""
        datetime_col = self.config.datetime_col
        target_col = self.config.target_col
        categorical_cols = self.config.categorical_cols
        drop_cols = self.config.drop_cols
        val_ratio = self.config.val_ratio
        scaler = SCALERS[self.config.scaler]
        scaler_path = self.config.scaler_path

        # Assert train, val, test ratios sum to 1
        assert np.isclose(self.config.train_ratio + self.config.val_ratio + self.config.test_ratio, 1.0)

        # Extract time features
        df = self.extract_time_features(df, datetime_col)

        # One-hot encode categorical variables
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
            logger.info(f'Applied one-hot encoding on {categorical_cols}.')

        # Get max train date
        max_train_date = self.get_max_train_date(df)
        if val_ratio >= VAL_RATIO_EPSILON:
            max_val_date = self.get_max_val_date(df)
        
         # Split into train/val/test sets
        train_df = df[df[self.config.datetime_col] <= max_train_date]
        X_train, y_train = self.get_X_y(train_df, df.columns, target_col)

        if val_ratio >= VAL_RATIO_EPSILON:
            val_df = df[(df[self.config.datetime_col] > max_train_date) & (df[self.config.datetime_col] <= max_val_date)]
            X_val, y_val = self.get_X_y(val_df, df.columns, target_col)
            # Get test set based on max val date
            test_df = df[(df[self.config.datetime_col] > max_val_date)]
            X_test, y_test = self.get_X_y(test_df, df.columns, target_col)
        
        else: 
            # Get test set based on max train date
            test_df = df[(df[self.config.datetime_col] > max_train_date)]
            X_test, y_test = self.get_X_y(test_df, df.columns, target_col)

        # Drop unnecessary columns
        X_train.drop(columns=[col for col in drop_cols if col in X_train.columns], inplace=True)
        X_test.drop(columns=[col for col in drop_cols if col in X_test.columns], inplace=True)
        if val_ratio >= VAL_RATIO_EPSILON:
            X_val.drop(columns=[col for col in drop_cols if col in X_val.columns], inplace=True)
        logger.info(f'Dropped columns: {drop_cols}.')

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        if val_ratio >= VAL_RATIO_EPSILON:
            X_val_scaled = scaler.transform(X_val)
        else: 
            X_val_scaled, y_val = None, None
        X_test_scaled = scaler.transform(X_test)
            
        self.save_scaler(scaler, scaler_path)
        logger.info('Completed feature scaling.')

        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)

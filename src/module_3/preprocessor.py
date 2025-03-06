import logging

import joblib
import numpy as np
import pandas as pd
from config import (
    CATEGORICAL_COLS,
    DATETIME_COL,
    DATETIME_DAY,
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
    datetime_day_col: str = DATETIME_DAY
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

    def filter_five_items_inside_order(self, df: pd.DataFrame) -> bool:
        order_size = df.groupby(self.config.order_col).outcome.sum()
        filtered_orders = order_size[order_size >= 5].index
        return df.loc[lambda x: x.order_id.isin(filtered_orders)]

    @staticmethod
    def extract_time_features(
        df: pd.DataFrame, datetime_col: str
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
    
    def get_max_train_date(self, df: pd.DataFrame, train_ratio) -> int:
        """Calculates the idx of the last user in the training set, ordered by date."""
        unique_orders = df.groupby(self.config.datetime_day_col)[self.config.order_col].nunique()
        cumsum_daily = unique_orders.cumsum() / unique_orders.sum()
        return unique_orders[cumsum_daily <= train_ratio].idxmax()
    
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
        datetime_day_col = self.config.datetime_day_col
        target_col = self.config.target_col
        categorical_cols = self.config.categorical_cols
        drop_cols = self.config.drop_cols
        train_ratio = self.config.train_ratio
        test_ratio = self.config.test_ratio
        val_ratio = self.config.val_ratio
        scaler = SCALERS[self.config.scaler]
        scaler_path = self.config.scaler_path

        # Assert train, val, test ratios sum to 1
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

        # Filter dataframe by specifications
        df = (
            df.pipe(self.filter_five_items_inside_order)
            .assign(created_at=lambda x:pd.to_datetime(x[datetime_col]))
            .assign(order_date=lambda x:pd.to_datetime(x[datetime_day_col]).dt.date)
        )
        logger.info('Filtered orders with more than 5 items.')

        # Extract time features
        # df = self.extract_time_features(df, datetime_col)

        # One-hot encode categorical variables
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
            logger.info(f'Applied one-hot encoding on {categorical_cols}.')

        # Get max train date
        max_train_date = self.get_max_train_date(df, train_ratio=train_ratio)
        if val_ratio >= VAL_RATIO_EPSILON:
            max_val_date = self.get_max_train_date(df, train_ratio=train_ratio + val_ratio)
        
         # Split into train/val/test sets
        feature_cols = [col for col in df.columns if col != target_col]
        train_df = df[df[datetime_day_col] <= max_train_date]
        X_train, y_train = self.get_X_y(train_df, df.columns, target_col)
        X_train, y_train = self.get_X_y(train_df, feature_cols, target_col)

        if val_ratio >= VAL_RATIO_EPSILON:
            val_df = df[(df[datetime_day_col] > max_train_date) & (df[datetime_day_col] <= max_val_date)]
            X_val, y_val = self.get_X_y(val_df, feature_cols, target_col)
            # Get test set based on max val date
            test_df = df[(df[datetime_day_col] > max_val_date)]
            X_test, y_test = self.get_X_y(test_df, feature_cols, target_col)
        
        else: 
            # Get test set based on max train date
            test_df = df[(df[datetime_day_col] > max_train_date)]
            X_test, y_test = self.get_X_y(test_df, feature_cols, target_col)

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

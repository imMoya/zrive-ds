import logging

import joblib
import numpy as np
import pandas as pd
from config import (
    CATEGORICAL_COLS,
    DATETIME_COL,
    DROP_COLS,
    ORDER_COL,
    PREPROCESSOR_CONFIG,
    SCALER,
    SCALER_PATH,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    USER_COL,
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


class PreprocessorConfig(BaseModel):
    datetime_col: str = DATETIME_COL
    user_col: str = USER_COL
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

    def chronological_user_split(
        self,
        df: pd.DataFrame,
        user_col: str,
        order_col: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits data chronologically per user."""
        if user_col not in df.columns or order_col not in df.columns:
            logger.error(
                f"Columns '{user_col}' or '{order_col}' not found in DataFrame."
            )
            raise ValueError('Missing required columns.')

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) <= 0.001, (
            'Ratios must sum to 1.'
        )

        train_list, val_list, test_list = [], [], []

        for _, user_data in df.groupby(user_col):
            user_data = user_data.sort_values(by=order_col)
            n = len(user_data)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_list.append(user_data.iloc[:train_end])
            val_list.append(user_data.iloc[train_end:val_end])
            test_list.append(user_data.iloc[val_end:])

        train_df, val_df, test_df = map(pd.concat, [train_list, val_list, test_list])
        logger.info('Completed chronological user split.')
        return train_df, val_df, test_df

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
        self, df: pd.DataFrame, config: dict = PREPROCESSOR_CONFIG
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Applies full preprocessing pipeline to the DataFrame."""
        datetime_col = config.get('datetime_col', 'created_at')
        user_col = config.get('user_col', 'user_id')
        order_col = config.get('order_col', 'user_order_seq')
        target_col = config.get('target_col', 'outcome')
        categorical_cols = config.get('categorical_cols', ['product_type'])
        drop_cols = config.get('drop_cols', ['created_at', 'order_date', 'vendor'])
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('val_ratio', 0.2)
        test_ratio = config.get('test_ratio', 0.1)
        scaler = config.get('scaler', StandardScaler())
        scaler_path = config.get('scaler_path', 'models/scaler.pkl')

        # Extract time features
        df = self.extract_time_features(df, datetime_col)

        # One-hot encode categorical variables
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
            logger.info(f'Applied one-hot encoding on {categorical_cols}.')

        # Drop unnecessary columns
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        logger.info(f'Dropped columns: {drop_cols}.')

        # Split into train/val/test sets
        train_df, val_df, test_df = self.chronological_user_split(
            df, user_col, order_col, train_ratio, val_ratio, test_ratio
        )

        # Extract features & target
        X_train, y_train = (
            train_df.drop(columns=[target_col]),
            train_df[target_col].values,
        )
        X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col].values
        X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col].values

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        self.save_scaler(scaler, scaler_path)
        logger.info('Completed feature scaling.')

        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)

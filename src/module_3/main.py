from pathlib import Path
from typing import Annotated
from logger import logger

import numpy as np
import pandas as pd
from config import PIPELINE_CONFIG
from dataloader import DataLoader
from preprocessor import Preprocessor, PreprocessorConfig
from pydantic import BaseModel
from zenml import pipeline, step
from zenml.integrations.numpy.materializers.numpy_materializer import NumpyMaterializer


class ProcessingParameters(BaseModel):
    data_path: str = PIPELINE_CONFIG['data_path']


@step
def load_data(params: ProcessingParameters) -> pd.DataFrame:
    data_loader = DataLoader(Path(params.data_path))
    return data_loader.load_data()


@step(
    output_materializers={
        'X_train_scaled': NumpyMaterializer,
        'y_train': NumpyMaterializer,
        'X_val_scaled': NumpyMaterializer,
        'y_val': NumpyMaterializer,
        'X_test_scaled': NumpyMaterializer,
        'y_test': NumpyMaterializer,
    }
)
def preprocess_data(
    df: pd.DataFrame,
) -> tuple[
    Annotated[np.ndarray, 'X_train_scaled'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'X_val_scaled'],
    Annotated[np.ndarray, 'y_val'],
    Annotated[np.ndarray, 'X_test_scaled'],
    Annotated[np.ndarray, 'y_test'],
]:
    preprocessor = Preprocessor(config=PreprocessorConfig())
    (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test) = (
        preprocessor.fit_transform(df)
    )
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


@pipeline
def training_pipeline(params: ProcessingParameters):
    df = load_data(params)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = (
        preprocess_data(df)
    )
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


if __name__ == '__main__':
    logger.info("Starting the pipeline")
    params = ProcessingParameters()
    training_pipeline(params)
    logger.info("Pipeline execution completed.")
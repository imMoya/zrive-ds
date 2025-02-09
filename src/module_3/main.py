from pathlib import Path
from typing import Annotated
import mlflow
import numpy as np
import pandas as pd
from config import DATA_PATH
from dataloader import DataLoader
from logger import logger
from preprocessor import Preprocessor, PreprocessorConfig
from pydantic import BaseModel
from model import Model, ModelConfig
from zenml import pipeline, step
from zenml.integrations.numpy.materializers.numpy_materializer import NumpyMaterializer


class ProcessingParameters(BaseModel):
    data_path: str | Path = DATA_PATH


@step
def load_data(params: ProcessingParameters) -> pd.DataFrame:
    data_loader = DataLoader(Path(params.data_path))
    return data_loader.load_data()


@step(
    output_materializers={
        'X_train': NumpyMaterializer,
        'y_train': NumpyMaterializer,
        'X_val': NumpyMaterializer,
        'y_val': NumpyMaterializer,
        'X_test': NumpyMaterializer,
        'y_test': NumpyMaterializer,
    }
)
def preprocess_data(
    df: pd.DataFrame,
) -> tuple[
    Annotated[np.ndarray, 'X_train'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'X_val'],
    Annotated[np.ndarray, 'y_val'],
    Annotated[np.ndarray, 'X_test'],
    Annotated[np.ndarray, 'y_test'],
]:
    preprocessor = Preprocessor(config=PreprocessorConfig())
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.fit_transform(
        df
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


@step
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    model = Model(config=ModelConfig())
    with mlflow.start_run() as run:
        model.train(X_train, y_train)
    logger.info("Model successfully trained.")
    return model

@step
def eval_model(model: Model, X_test: np.ndarray, y_test: np.ndarray):
    mse = model.evaluate(X_test, y_test)
    logger.info("Model evaluated.")
    return mse

@step
def save_model_with_mlflow(model: Model):
    with mlflow.start_run():
        logger.info("Model saved in MLflow.")


@step
def log_metrics_with_mlflow(mse: float):
    with mlflow.start_run():
        mlflow.log_metric("mean_squared_error", mse)
        logger.info(f"Logged MSE to MLflow: {mse}")


@pipeline
def training_pipeline(params: ProcessingParameters):
    # Load data
    df = load_data(params)
    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(df)
    # Train model
    model = train_model(X_train, y_train)
    # Evaluate model
    mse = eval_model(model, X_test, y_test)
    # Save model
    save_model_with_mlflow(model)
    # Log metrics
    log_metrics_with_mlflow(mse)
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    logger.info('Starting the pipeline')
    params = ProcessingParameters()
    training_pipeline(params)
    # TODO: Check mlflow https://github.com/mlflow/mlflow/blob/master/examples/sklearn_logistic_regression/train.pyhttps://github.com/mlflow/mlflow/blob/master/examples/sklearn_logistic_regression/train.py
    logger.info('Pipeline execution completed.')

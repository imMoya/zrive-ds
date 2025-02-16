import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Annotated, Optional
import mlflow
import numpy as np
import pandas as pd
from config import DATA_PATH
from dataloader import DataLoader
from logger import logger
from preprocessor import Preprocessor, PreprocessorConfig
from pydantic import BaseModel
from model import Model, ModelConfig, Metrics, compute_and_log_metrics
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
    X_val = X_val if X_val is not None else np.empty((0, df.shape[1]))
    y_val = y_val if y_val is not None else np.empty((0,))
    return X_train, y_train, X_val, y_val, X_test, y_test


@step
def train_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple[Model, Metrics]:
    model = Model(config=ModelConfig())
    with mlflow.start_run() as run:
        # Train model
        model.train(X_train, y_train)
        logger.info('Model successfully trained.')

        # Save model
        mlflow.sklearn.log_model(model.model, "model")
        logger.info('Model saved in MLflow.')

        # Log metrics
        train_metrics = compute_and_log_metrics(y_train, model.model.predict_proba(X_train)[:, 1])
        test_metrics = compute_and_log_metrics(y_test, model.model.predict_proba(X_test)[:, 1])

        prefix = "train_"
        mlflow.log_metric(f'{prefix}precision_recall_auc', train_metrics.pr_auc)
        mlflow.log_metric(f'{prefix}roc_auc', train_metrics.roc_auc)
        prefix = "test_"
        mlflow.log_metric(f'{prefix}precision_recall_auc', test_metrics.pr_auc)
        mlflow.log_metric(f'{prefix}roc_auc', test_metrics.roc_auc)
        mlflow.log_param(f'{prefix}fpr', json.dumps(test_metrics.fpr.tolist()))
        mlflow.log_param(f'{prefix}tpr', json.dumps(test_metrics.tpr.tolist()))
        mlflow.log_param(f'{prefix}precision', json.dumps(test_metrics.precision.tolist()))
        mlflow.log_param(f'{prefix}recall', json.dumps(test_metrics.recall.tolist()))
    
    return model, test_metrics

@step
def plot_metrics(model: Model, test_metrics: Metrics):
    def plot(metrics: Metrics, model_id: str, figure: tuple[plt.Figure, np.array] = None) -> tuple[plt.Figure, np.array]:
        if figure is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        else:
            fig, ax = figure
        
        ax[0].plot(metrics.recall, metrics.precision, label=f"{model_id} AUC:{metrics.pr_auc:.2f}")
        ax[0].set_xlabel("Recall")
        ax[0].set_ylabel("Precision")
        ax[0].set_title("Precision-Recall Curve")
        ax[0].legend()
        
        
        ax[1].plot(metrics.fpr, metrics.tpr, label=f"{model_id} AUC:{metrics.roc_auc:.2f}")
        ax[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].set_title("ROC Curve")
        ax[1].legend()
        
        return fig, ax
    
    fig, ax = plot(test_metrics, model.config.model_name)
    plt.show()



@pipeline
def training_pipeline(params: ProcessingParameters):
    # Load data
    df = load_data(params)
    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(df)
    # Train model, save model and log metrics
    model, metrics = train_model(X_train, y_train, X_val, y_val, X_test, y_test)
    # Plot model vs baseline and best performing model
    plot_metrics(model, metrics)
    return model, metrics



if __name__ == '__main__':
    logger.info('Starting the pipeline')
    mlflow.set_experiment("test")
    params = ProcessingParameters()
    training_pipeline(params)
    logger.info('Pipeline execution completed.')
    # TODO: Create a step to select a set of model and plot them based on the MLFlow outputs


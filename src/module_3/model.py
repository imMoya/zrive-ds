import numpy as np
from config import RANDOM_STATE
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from pydantic import BaseModel, Field
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass

class ModelConfig(BaseModel):
    model_name: str = 'logistic_regression'
    random_state: int = RANDOM_STATE
    hyperparameters: dict = Field(default_factory=dict)


class Model:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        model_class = {
            'linear_regression': LinearRegression,
            'lasso': Lasso,
            'ridge': Ridge,
            'elastic_net': ElasticNet,
            'logistic_regression': LogisticRegression
        }.get(self.config.model_name)

        if model_class is None:
            raise ValueError(f'Unsupported model: {self.config.model_name}')

        return model_class(**self.config.hyperparameters)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self.model

@dataclass
class Metrics:
    precision: np.ndarray
    recall: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    pr_auc: float
    roc_auc: float

def compute_and_log_metrics(y_true, y_pred, prefix=""):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return Metrics(
        precision=precision,
        recall=recall,
        fpr=fpr,
        tpr=tpr,
        pr_auc=pr_auc,
        roc_auc=roc_auc
    )


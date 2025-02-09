import numpy as np
from config import RANDOM_STATE
from pydantic import BaseModel, Field
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


class ModelConfig(BaseModel):
    model_name: str = 'linear_regression'
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
        }.get(self.config.model_name)

        if model_class is None:
            raise ValueError(f'Unsupported model: {self.config.model_name}')

        return model_class(**self.config.hyperparameters)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse


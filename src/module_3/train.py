import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


class TrainerConfig(BaseModel):
    model_name: str = 'linear_regression'
    random_state: int = 42


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.models = {
            'linear_regression': LinearRegression(),
            'lasso': Lasso(),
            'ridge': Ridge(),
            'elastic_net': ElasticNet(),
        }
        self.model = self._initialize_model()

    def _initialize_model(self):
        try:
            return self.models[self.config.model_name]
        except KeyError as e:
            raise ValueError(f'Unsupported model: {self.config.model_name}') from e

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

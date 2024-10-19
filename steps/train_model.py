import logging
import pandas as pd

from zenml import step
from src.model_dev import RandomForestClassifierModel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig
    ) -> ClassifierMixin:
    """__summary__
    Train the model using the ingested data.
    
    Args:
        X_train: Training data
        y_train: Training target
    """
    try:
        if config.model_name == 'RandomForestClassifier':
            mlflow.sklearn.autolog()
            model = RandomForestClassifierModel()
            model = model.train(X_train, y_train)
            return model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error while training model: {str(e)}")
        raise e
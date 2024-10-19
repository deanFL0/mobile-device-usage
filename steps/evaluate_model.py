import logging
import pandas as pd
from zenml import step
from sklearn.base import ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import Accuracy, Precision, Recall, F1Score

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                    X_test: pd.DataFrame,
                    y_test: pd.Series
                    ) -> Tuple[
                        Annotated[float, 'accuracy'],
                        Annotated[float, 'precision'],
                        Annotated[float, 'recall'],
                        Annotated[float, 'f1_score']
                    ]:
    """
    Evaluate the model using the test data.
    
    Args:
        df: pandas dataframe
    """
    try:
        predictions = model.predict(X_test)
        accuracy = Accuracy().calculate_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        precision = Precision().calculate_score(y_test, predictions)
        mlflow.log_metric("precision", precision)
        recall = Recall().calculate_score(y_test, predictions)
        mlflow.log_metric("recall", recall)
        f1_score = F1Score().calculate_score(y_test, predictions)
        mlflow.log_metric("f1_score", f1_score)
        return accuracy, precision, recall, f1_score
    except Exception as e:
        logging.error(f"Error in evaluating model: {str(e)}")
        raise e
    
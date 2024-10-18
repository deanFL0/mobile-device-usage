import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating model
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """__summary__
        calculate score
        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Predicted Labels
        """
        pass

class Accuracy(Evaluation):
    """
    Calculate accuracy score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """__summary__
        calculate accuracy score
        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Predicted Labels
        Returns:
            float: accuracy score
        """
        try:
            logging.info("Calculating accuracy score")
            accuracy = accuracy_score(y_true, y_pred)
            return accuracy
        except Exception as e:
            logging.error(f"Error in calculating accuracy score: {str(e)}")
            raise e
    
class Precision(Evaluation):
    """
    Calculate precision score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """__summary__
        calculate precision score
        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Predicted Labels
        Returns:
            float: precision score
        """
        try:
            logging.info("Calculating precision score")
            return precision_score(y_true, y_pred, average='macro')
        except Exception as e:
            logging.error(f"Error in calculating precision score: {str(e)}")
            raise e
        
class Recall(Evaluation):
    """
    Calculate recall score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """__summary__
        calculate recall score
        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Predicted Labels
        Returns:
            float: recall score
        """
        try:
            logging.info("Calculating recall score")
            return recall_score(y_true, y_pred, average='macro')
        except Exception as e:
            logging.error(f"Error in calculating recall score: {str(e)}")
            raise e

class F1Score(Evaluation):
    """
    Calculate f1 score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """__summary__
        calculate f1 score
        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Predicted Labels
        Returns:
            float: f1 score
        """
        try:
            logging.info("Calculating f1 score")
            return f1_score(y_true, y_pred, average='macro')
        except Exception as e:
            logging.error(f"Error in calculating f1 score: {str(e)}")
            raise e
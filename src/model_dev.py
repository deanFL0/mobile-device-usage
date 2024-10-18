import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model
        Args:
            X_train: Tranining data
            y_train: Training target
        """
        pass
    
class RandomForestClassifierModel(Model):
    """
    Random Forest Classifier model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model
        Args:
            X_train: Training data
            y_train: Training target
        """
        try:
            model = RandomForestClassifier(**kwargs)
            model.fit(X_train, y_train)
            logging.info(f'Model training completed')
            return model 
        except Exception as e:
            logging.error(f'Error while training model: {e}')
            raise e
        
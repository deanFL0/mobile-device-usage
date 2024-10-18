import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreprocessStrategy(DataStrategy):
    """
    Strategy to preprocess data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        preprocess data
        """
        try:
            data = data.drop(['Device Model', 'User ID'], axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {str(e)}")
            raise e
        
class DataSplitStrategy(DataStrategy):
    """
    Strategy to split data
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        split data
        """
        try:
            X = data.drop(['User Behavior Class'], axis=1)
            y = data['User Behavior Class']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {str(e)}")
            raise e    
        
class DataEncodingStrategy:
    """
    Strategy to encode data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode data
        """
        try:
            data_original = data.copy()
            data = data.drop(['User Behavior Class'], axis=1)
            cat_cols = data.select_dtypes(include=['object']).columns
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            data_encoded = encoder.fit_transform(data[cat_cols])
            data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(cat_cols))
            data = pd.concat([data, data_encoded_df], axis=1)
            data = data.drop(cat_cols, axis=1)
            data['User Behavior Class'] = data_original['User Behavior Class']
            return data
        except Exception as e:
            logging.error(f"Error in encoding data: {str(e)}")
            raise e
        
class DataScalingStrategy:
    """
    Strategy to scale data
    """
    def handle_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale data
        """
        try:
            scaler = MinMaxScaler()
            train_data_scaled = scaler.fit_transform(train_data)
            test_data_scaled = scaler.transform(test_data)
            train_data = pd.DataFrame(train_data_scaled, columns=train_data.columns)
            test_data = pd.DataFrame(test_data_scaled, columns=test_data.columns)
            return train_data, test_data
        except Exception as e:
            logging.error(f"Error in scaling data: {str(e)}")
            raise e
class DataPreprocessing:
    """
    Class to handle data preprocessing
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy, test_data: pd.DataFrame = None):
        self.data = data
        self.test_data = test_data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data using strategy
        """
        try:
            if self.test_data is not None:
                # Your logic for handling both dataframes
                return self.strategy.handle_data(self.data, self.test_data)
            else:
                return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {str(e)}")
            raise e
            
            
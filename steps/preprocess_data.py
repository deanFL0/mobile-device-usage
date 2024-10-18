import logging
import pandas as pd
from zenml import step
from src.data_preprocess import DataPreprocessStrategy, DataSplitStrategy, DataEncodingStrategy, DataScalingStrategy, DataPreprocessing
from typing_extensions import Annotated
from typing import Tuple

@step
def preprocess_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    """
    Clean the data and split it into train and test set
    
    Args:
        df: Raw data
    Return:
        X_train: Train data
        X_test: Test data
        y_train: Train target
        y_test: Test target
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_preprocess = DataPreprocessing(df, preprocess_strategy)
        data = data_preprocess.handle_data()
        
        encoding_strategy = DataEncodingStrategy()
        data_encoding = DataPreprocessing(data, encoding_strategy)
        data = data_encoding.handle_data()
        
        split_strategy = DataSplitStrategy()
        data_split = DataPreprocessing(data, split_strategy)
        X_train, X_test, y_train, y_test = data_split.handle_data()
        
        scaling_strategy = DataScalingStrategy()
        data_scaling = DataPreprocessing(X_train, scaling_strategy, X_test)
        X_train, X_test = data_scaling.handle_data()
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in preprocessing data: {str(e)}")
        raise e
        
        
        
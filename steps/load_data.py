import logging
import pandas as pd
from zenml import step

class LoadData:
    '''
    Load data from a data_path
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path

    def run(self):
        logging.info(f"Loading data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def load_data(data_path: str) -> pd.DataFrame:
    '''
    Load data from a data_path
    
    Args:
        data_path: str: Path to the data
    Returns:
        pd.DataFrame: Dataframe containing the data
    '''
    try:
        loader = LoadData(data_path)
        return loader.run()
    except Exception as e:
        logging.error(f"Failed to load data from {data_path}")
        logging.error(e)
        raise e
        
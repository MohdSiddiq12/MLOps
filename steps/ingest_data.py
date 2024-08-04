import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self,data_path:str):
        self.data_path = data_path

    def get_data(self):
        logging.info("ingesting data fron {self.data_path}")
        return pd.read_csv(self.data_path)
        
@step
def ingest_df(data_path: str ) -> pd.DataFrame:

    """
    Ingests data from the given path and returns a pandas DataFrame.

    Args:
        data_path (str): The path to the data file.

    returns:
        pd.DataFrame: A pandas DataFrame containing the data. 
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
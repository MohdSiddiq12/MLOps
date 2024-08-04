import logging
import pandas as pd
from zenml.steps import step


@step

def train_model(df: pd.DataFrame) -> None:
    """
    Trains a model on the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to train the model on.

    """
    pass
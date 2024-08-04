import logging
import pandas as pd
from zenml.steps import step


@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluates a model on the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to evaluate the model on.

    """
    pass
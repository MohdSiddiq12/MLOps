from zenml import pipelines
from steps.ingest_df import ingest_df
from steps.clean_df import clean_df
from steps.evaluation import evaluate_model
from steps.model_train import train_model

@pipeline()
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)

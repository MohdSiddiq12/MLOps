from pipelines.training_pipeline import train_pipeline
if __name__ == "__main__":
    #run the pipeline
    train_pipeline(data_path="data/raw/housing.csv")
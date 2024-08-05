import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import joblib
# Load data
data = pd.read_csv('data/raw/housing.csv')
X = data.drop(['median_house_value','ocean_proximity'], axis=1)
y = data['median_house_value']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name
experiment_name = "version 1.3"

# Create a new experiment
mlflow.create_experiment(experiment_name)

# Set the new experiment as the active one
mlflow.set_experiment(experiment_name)
# Start an MLflow run
with mlflow.start_run():
    # Set model parameters
    n_estimators = 100
    max_depth = 10
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Log metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    #add more model metrics like mae,r2 etc
    signature = infer_signature(X_test,model.predict(X_test))
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)

       # Register the model
    # Register the model with version 1.1, tags, and aliases
    register = mlflow.register_model(
        "runs:/9f3fba9faf4345dfbd3ccc65a2fc15c0/random_forest_model",
        "RandomForestModel",
        tags={"registering-model": "true", "model-type": "regressor"}
    )

    # Add aliases to the registered model
    client = mlflow.tracking.MlflowClient()
    client.update_model_version(
        name="RandomForestModel",
        version=register.version,
        description="Model version 1.1"
    )
    client.add_model_version_tag(
        name="RandomForestModel",
        version=register.version,
        tag="alias", value="production"
    )
    

    print(f"Logged model with MSE: {mse}, MAE: {mae}, R2: {r2}")

print("Training complete.")

#  # Save model to file 
# joblib.dump(model, 'RandomForest.pkl')
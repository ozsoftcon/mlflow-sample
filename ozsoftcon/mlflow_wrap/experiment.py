from typing import Dict
from mlflow.client import MlflowClient


def create_experiment(
        mlflow_client: MlflowClient,
        experiment_name: str
) -> Dict[str, str]:
    existing_experiment = mlflow_client.get_experiment_by_name(
        experiment_name
    )

    if existing_experiment is not None:
        print("Experiment already exists. Returning previous id")
        return existing_experiment.experiment_id

    new_id = mlflow_client.create_experiment(
        name=experiment_name
    )

    print("New experiment created. Returing new id")
    return new_id

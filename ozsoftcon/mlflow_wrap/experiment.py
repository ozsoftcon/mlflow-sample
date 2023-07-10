from typing import Dict, Optional, Any
from uuid import uuid4
from mlflow import set_experiment, end_run
from mlflow.entities import Run
from mlflow.client import MlflowClient

def create_experiment(
        mlflow_client: MlflowClient,
        experiment_name: str
) -> str:
    existing_experiment = mlflow_client.get_experiment_by_name(
        experiment_name
    )

    if existing_experiment is not None:
        print("Experiment already exists. Returning previous id")
        return existing_experiment.experiment_id

    new_id = mlflow_client.create_experiment(
        name=experiment_name
    )

    print("New experiment created. Returning new id")
    return new_id


def create_run_in_experiment(
        mlflow_client: MlflowClient,
        experiment_id: str,
        tags: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None
) -> Run:
    if not run_name:
        run_name = f"Run {uuid4().hex} created for experiment"
    set_experiment(experiment_id=experiment_id)
    new_run = mlflow_client.create_run(
        experiment_id=experiment_id, tags=tags, run_name=run_name
    )

    return new_run
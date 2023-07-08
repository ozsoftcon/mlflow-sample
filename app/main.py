from os import environ
from dotenv import load_dotenv
from mlflow import log_param, ActiveRun
from ozsoftcon.mlflow_wrap import (
    MLFlowConfig, create_experiment
)

load_dotenv()

tracking_uri = environ.get("MLFLOW_TRACKING_URI", "")
registry_uri = environ.get("MLFLOW_REGISTRY_URI", "")


def main():

    mlflow_config = MLFlowConfig(tracking_uri, registry_uri)
    print("Creating New Experiment")
    experiment_id = create_experiment(
        mlflow_config.mlflow_client,
        "sample_experiment3"
    )

    current_run = mlflow_config.create_run_for_experiment(
        experiment_id,
        tags={"test": "test"},
        run_name="run_name"
    )
    with ActiveRun(current_run) as run:
        log_param("test-value", 10)


if __name__ == "__main__":
    main()

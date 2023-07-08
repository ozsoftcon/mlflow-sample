from os import environ
from dotenv import load_dotenv
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
        "sample_experiment2"
    )
    print(f"New Experiment ID: {experiment_id}")
    print("Reusing Old Experiment")
    experiment_id = create_experiment(
        mlflow_config.mlflow_client,
        "sample_experiment2"
    )
    print(f"Reusing Experiment Id {experiment_id}")


if __name__ == "__main__":
    main()

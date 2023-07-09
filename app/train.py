from os import environ
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score
from dotenv import load_dotenv
from mlflow import log_param, ActiveRun
from ozsoftcon.mlflow_wrap import (
    MLFlowConfig, create_experiment
)
from ozsoftcon.ml import read_ml_data

load_dotenv()

tracking_uri = environ.get("MLFLOW_TRACKING_URI", "")
registry_uri = environ.get("MLFLOW_REGISTRY_URI", "")


def main():

    # mlflow_config = MLFlowConfig(tracking_uri, registry_uri)
    # print("Creating New Experiment")
    # experiment_id = create_experiment(
    #     mlflow_config.mlflow_client,
    #     "sample_experiment3"
    # )
    #
    # current_run = mlflow_config.create_run_for_experiment(
    #     experiment_id,
    #     tags={"test": "test"},
    #     run_name="run_name"
    # )
    # with ActiveRun(current_run) as run:
    #     log_param("test-value", 10)

    test_fraction = 0.1
    validation_fraction = 0.1
    train_data, validation_data, _ = read_ml_data(
        "./sample_data/data.csv",
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        seed_value=142
    )

    model_parameters = {
        "n_clusters": 4,
        "tol": 1e-10
    }

    train_features = train_data[:, 0:2]
    train_labels = train_data[:, 2]
    validation_features = validation_data[:, 0:2]
    validation_labels = validation_data[:, 2]
    model = KMeans(**model_parameters)
    model.fit(train_features, train_labels)

    prediction = model.predict(validation_features)
    adjusted_mi_score = adjusted_mutual_info_score(validation_labels, prediction)
    v_measure = v_measure_score(validation_labels, prediction)

    print(max(prediction))
    print(validation_labels[0:10])

    print(f"Adjusted mutual informatio score {adjusted_mi_score}")
    print(f"V Measure {v_measure}")

if __name__ == "__main__":
    main()

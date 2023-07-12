from os import environ
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score
from dotenv import load_dotenv
from mlflow import log_params, start_run, log_metrics, set_experiment, log_artifact
from mlflow.sklearn import log_model, SERIALIZATION_FORMAT_CLOUDPICKLE
from ozsoftcon.mlflow_wrap import (
    MLFlowConfig, create_experiment, create_run_in_experiment
)
from ozsoftcon.ml import read_ml_data

load_dotenv()

tracking_uri = environ.get("MLFLOW_TRACKING_URI", "")
registry_uri = environ.get("MLFLOW_REGISTRY_URI", "")


def main():

    mlflow_config = MLFlowConfig(tracking_uri, registry_uri)
    print("Creating New Experiment")
    experiment_id = create_experiment(
        mlflow_config.mlflow_client,
        "Simple Clustering Problem"
    )
    set_experiment(experiment_id=experiment_id)

    current_run = create_run_in_experiment(
        mlflow_config.mlflow_client,
        experiment_id,
        tags={"test": "test"},
        run_name="Try with clusters 4"
    )

    with start_run(run_id=current_run.info.run_id) as run:
        test_fraction = 0.1
        validation_fraction = 0.1
        data_fold_seed = 142
        train_data, validation_data, _ = read_ml_data(
            "./sample_data/data.csv",
            test_fraction=test_fraction,
            validation_fraction=validation_fraction,
            seed_value=data_fold_seed
        )

        training_parameters = {
            "data_fold_seed_value": data_fold_seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "algorithm": "KMeans"
        }

        log_params(training_parameters)

        model_parameters = {
            "n_clusters": 4
        }
        log_params(model_parameters)

        train_features = train_data[:, 0:2]
        train_labels = train_data[:, 2]
        validation_features = validation_data[:, 0:2]
        validation_labels = validation_data[:, 2]
        model = KMeans(**model_parameters)
        model.fit(train_features, train_labels)

        prediction = model.predict(validation_features)
        adjusted_mi_score = adjusted_mutual_info_score(validation_labels, prediction)
        v_measure = v_measure_score(validation_labels, prediction)

        validation_metrics = {
            "adjusted_mi_score": adjusted_mi_score,
            "v_measure": v_measure
        }

        log_metrics(validation_metrics)

        log_model(
            sk_model=model,
            artifact_path="clustering_model",
            conda_env="conda_env.yaml",
            serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name="simple_clustering_model"
        )


if __name__ == "__main__":
    main()

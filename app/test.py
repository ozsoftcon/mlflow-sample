from os import environ
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score
from dotenv import load_dotenv
from mlflow import get_run
from mlflow.sklearn import load_model
from ozsoftcon.ml import read_ml_data

load_dotenv()

tracking_uri = environ.get("MLFLOW_TRACKING_URI", "")
registry_uri = environ.get("MLFLOW_REGISTRY_URI", "")

CHOSEN_RUN_ID = "60141d9dad9c45788d8522678b77b107"
run_uri = f"runs:/{CHOSEN_RUN_ID}/clustering_model"

def main():

    # we need to reload the test data using exact parameters we used
    # when we loaded the data during training; luckily we saved it during the
    # run as parameter
    chosen_run = get_run(run_id=CHOSEN_RUN_ID)
    params = chosen_run.data.params
    source_data = params["source_data"]
    data_fold_seed = int(params["data_fold_seed_value"])
    test_fraction = float(params["test_fraction"])
    validation_fraction = float(params["validation_fraction"])

    _ , test_data, _ = read_ml_data(
            source_data,
            test_fraction=test_fraction,
            validation_fraction=validation_fraction,
            seed_value=data_fold_seed
        )

    cluster_model = load_model(run_uri)

    test_features = test_data[:, 0:2]
    test_labels = test_data[:, 2]
    prediction = cluster_model.predict(test_features)
    adjusted_mi_score = adjusted_mutual_info_score(test_labels, prediction)
    v_measure = v_measure_score(test_labels, prediction)

    test_metrics = {
        "adjusted_mi_score": adjusted_mi_score,
        "v_measure": v_measure
    }

    print(f"Performance on test data: {test_metrics}")



if __name__ == "__main__":
    main()

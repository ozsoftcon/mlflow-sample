from unittest import TestCase, mock
from mlflow import set_experiment, end_run
from ozsoftcon.mlflow_wrap import MLFlowConfig
from ozsoftcon.mlflow_wrap import create_experiment, create_run_in_experiment


class TestExperiment(TestCase):

    def test_create_new_experiment(
            self
    ):
        class MockExperiment:

            def __init__(self):
                self.experiment_id = 1

        class MockMlFlowClient:

            def get_experiment_by_name(
                    self,
                    experiment_name):
                if experiment_name == "existing":
                    return MockExperiment()
                else:
                    return None

            def create_experiment(self, name=""):
                if name == "new":
                    return 2

        mlflow_client = MockMlFlowClient()

        experiment_id = create_experiment(
            mlflow_client, "existing"
        )
        assert experiment_id==1

        experiment_id = create_experiment(
            mlflow_client, "new"
        )
        assert experiment_id==2

    def test_create_valid_run(
            self
    ):
        mlflow_config = MLFlowConfig(
            "valid-tracking-uri",
            "valid-registry-uri"
        )

        experiment_id = create_experiment(
            mlflow_config.mlflow_client, "test-experiment"
        )
        set_experiment(experiment_id=experiment_id)
        test_run = create_run_in_experiment(
            mlflow_config.mlflow_client,
            experiment_id=experiment_id,
            tags={"subject": "cifar-10"},
            run_name="test-run"
        )

        assert test_run.info.run_name == "test-run"
        assert test_run.data.tags["mlflow.runName"] == "test-run"
        assert test_run.data.tags["subject"] == "cifar-10"
        assert test_run.info.experiment_id == experiment_id

        end_run(status="FINISHED")

    @mock.patch('ozsoftcon.mlflow_wrap.experiment.uuid4')
    def test_create_run_with_no_name(
            self, mocked_uuid4
    ):
        class MockedUUID4:
            hex = "123456"

        mocked_uuid4.return_value = MockedUUID4()

        mlflow_config = MLFlowConfig(
            "valid-tracking-uri",
            "valid-registry-uri"
        )

        experiment_id = create_experiment(
            mlflow_config.mlflow_client, "test-experiment")
        set_experiment(experiment_id=experiment_id)

        test_run = create_run_in_experiment(
            mlflow_config.mlflow_client,
            experiment_id, {"subject": "cifar-10"}
        )
        assert test_run.info.run_name == "Run 123456 created for experiment"
        assert test_run.data.tags["mlflow.runName"] == "Run 123456 created for experiment"
        assert test_run.data.tags["subject"] == "cifar-10"
        assert test_run.info.experiment_id == experiment_id
        end_run(status="FINISHED")

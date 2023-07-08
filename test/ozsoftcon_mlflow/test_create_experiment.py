from unittest import TestCase, mock

from ozsoftcon.mlflow_wrap import create_experiment

class TestCreateExperiment(TestCase):

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
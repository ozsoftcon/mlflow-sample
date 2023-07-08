from pytest import raises
from unittest import mock, TestCase
from ozsoftcon.mlflow_wrap import MLFlowConfig
from ozsoftcon.mlflow_wrap import create_experiment
from ozsoftcon.utils import InvalidMLFlowUri

class TestMlFlowConfig(TestCase):

    def test_create_invalid_config(self):

        with raises(InvalidMLFlowUri):
            with mock.patch.dict(
                'os.environ', {
                        'MLFLOW_TRACKING_URI': '',
                        'MLFLOW_REGISTRY_URI': ''
                }
            ):
                _ = MLFlowConfig()

    @mock.patch('ozsoftcon.mlflow_wrap.mlflow_config.set_tracking_uri')
    @mock.patch('ozsoftcon.mlflow_wrap.mlflow_config.set_registry_uri')
    def test_create_valid_config(
            self,
            mocked_registry_uri_set,
            mocked_tracking_uri_set
    ):

        mlflow_config = MLFlowConfig('valid-tracking-uri', 'valid-registry-uri')
        assert mlflow_config.mlflow_client.tracking_uri == 'valid-tracking-uri'
        assert mocked_tracking_uri_set.called_once_with('valid-tracking-uri')
        assert mocked_registry_uri_set.called_once_with('valid-registry-uri')


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

        test_run = mlflow_config.create_run_for_experiment(
            experiment_id, {"subject": "cifar-10"}, "test-run"
        )

        assert test_run.info.run_name == "test-run"
        self.assertDictEqual(
            test_run.data.tags, {
                "mlflow.runName": "test-run",
                "subject": "cifar-10"
            })
        assert test_run.info.experiment_id == experiment_id

    @mock.patch('ozsoftcon.mlflow_wrap.mlflow_config.uuid4')
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

        test_run = mlflow_config.create_run_for_experiment(
            experiment_id, {"subject": "cifar-10"}
        )
        assert test_run.info.run_name == "Run 123456 created for experiment"
        self.assertDictEqual(
            test_run.data.tags, {
                "mlflow.runName": "Run 123456 created for experiment",
                "subject": "cifar-10"
            })
        assert test_run.info.experiment_id == experiment_id

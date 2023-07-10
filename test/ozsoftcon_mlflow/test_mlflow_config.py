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

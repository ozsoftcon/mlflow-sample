from pytest import raises
from unittest import mock, TestCase
from ozsoftcon.mlflow_wrap import MLFlowConfig
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

        with mock.patch.dict(
            'os.environ', {
                    'MLFLOW_TRACKING_URI': 'valid-tracking-uri',
                    'MLFLOW_REGISTRY_URI': 'valid-registry-uri'
            }
        ):
            mlflow_config = MLFlowConfig()
            assert mlflow_config.mlflow_client.tracking_uri == 'valid-tracking-uri'
            assert mocked_tracking_uri_set.called_once_with('valid-tracking-uri')


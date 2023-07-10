from dotenv import load_dotenv
from mlflow.client import MlflowClient
from mlflow import set_tracking_uri, set_registry_uri, ActiveRun, start_run

from ..utils import InvalidMLFlowUri

load_dotenv()


class MLFlowConfig():

    def __init__(
            self,
            tracking_uri: str = "",
            registry_uri: str = ""
    ):

        if tracking_uri == "" or registry_uri == "":
            raise InvalidMLFlowUri(
                tracking_uri,
                registry_uri
            )

        set_tracking_uri(tracking_uri)
        set_registry_uri(registry_uri)

        self.mlflow_client = MlflowClient(
            tracking_uri,
            registry_uri
        )




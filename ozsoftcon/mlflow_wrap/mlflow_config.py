from typing import Optional, Dict, Any
from uuid import uuid4
from dotenv import load_dotenv
from mlflow.client import MlflowClient
from mlflow import set_tracking_uri, set_registry_uri
from mlflow.entities import Run

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

    def create_run_for_experiment(
            self,
            experiment_id: str,
            tags: Optional[Dict[str, Any]] = None,
            run_name: Optional[str] = None
    ) -> Run:
        if not run_name:
            run_name = f"Run {uuid4().hex} created for experiment"
        new_run = self.mlflow_client.create_run(
            experiment_id, tags=tags, run_name=run_name
        )

        return new_run



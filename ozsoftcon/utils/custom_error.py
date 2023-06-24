from typing import Dict


class CustomException(Exception):

    def __init__(
            self,
            params: Dict[str, str] ={},
            prefix_string: str = "",
            suffix_string: str = ""
    ):
        if params:
            self.msg = f"{prefix_string};{params};{suffix_string}"
        else:
            self.msg = f"{prefix_string};{suffix_string}"
        super().__init__(self.msg)


class InvalidMLFlowUri(CustomException):

    def __init__(
            self,
            tracking_uri: str,
            registry_uri: str
    ):
        params = {
            "tracking_uri": tracking_uri,
            "registry_uri": registry_uri
        }

        super().__init__(
            params,
            "Invalid MLFlow Uri provided",
            "Please provide valid Uris"
        )
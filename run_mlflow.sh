#!/bin/bash

mlflow server --backend-store-uri mysql+pymysql://$MLFLOW_DB_USER:$MLFLOW_DB_PWD@127.0.0.1:3306/mlflow_database --default-artifact-root $MLFLOW_ARTIFACT_PATH --host 0.0.0.0 --port 5000

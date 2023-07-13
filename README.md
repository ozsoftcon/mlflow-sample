# Using MLFlow: A Sample Codebase

## Setup

1. If you do not have `conda`, download and install `conda` following installer and instruction suitable to your system
from [here](https://docs.conda.io/en/main/miniconda.html#latest-miniconda-installer-links).

2. Run the following in terminal:
```commandline
$ conda create -n mlflow-sample python==3.10.11
$ conda activate mlflow-sample
$ pip install -r requirements.txt
$ pip install -r requirements-dev.txt
```

3. Clone the repository
```commandline
$ git clone git@github.com:ozsoftcon/mlflow-sample.git
```

4. Change to the code folder and install the package

```commandline
$ cd mlflow-sample
$ pip install .
```

5. Run the unit test
```commandline
$ pytest test/test_**
```

If the unit tests run successfully, we are good to go.

## Preparing and running the MLFlow Server

```commandline
$ cd ./mlflow-data
$ docker compose up -d --build
```
This will start the MySQL server as backend for the MLFlow.

NOTE: If you already have `MySql` running on your system, you have to stop that.
```commandline
$ sudo service mysql stop
```

Otherwise, you can modify the `docker-componse.yml` to use a different port.

Finally, start the MLFlow server:

```commandline
$ export MLFLOW_DB_USER=admin
$ export MLFLOW_DB_PASSWORD=1234#567
$ export MLFLOW_ARTIFACT_PATH=/add/any/wrtable/path/
$ mlflow server --backend-store-uri mysql+pymysql://$MLFLOW_DB_USER:$MLFLOW_DB_PWD@127.0.0.1:3306/mlflow_database --default-artifact-root $MLFLOW_ARTIFACT_PATH --host 0.0.0.0 --port 5000
```

Once the server starts, you can browse the server at `http://127.0.0.1:5000`.
![MLFlowService](./images/mlflow_server.png)

Associated Medium Article: https://medium.com/@rajib.chakravorty/introduction-to-mlflow-with-complete-code-25b6ced6d32f



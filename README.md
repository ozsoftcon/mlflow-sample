# Using MLFlow: A Sample Codebase

## Setup

1. If you do not have `conda`, download and install `conda` following installer and instruction suitable to your system
from [here](https://docs.conda.io/en/main/miniconda.html#latest-miniconda-installer-links).

2. Run the following in terminal:
```commandline
$ conda create -n mlflow-env python==3.10.11
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

## Preparing and running the docker

```commandline
$ cd ./mlflow-docker
$ docker compose up -d --build
```

Once the build finishes, you can browse the server at `http://127.0.0.1:5000`.
![MLFlowService](./images/mlflow_server.png)

NOTE: If you already have `MYSql` running on your system, you have to stop that.
On Ubuntu you can try
```commandline
$ sudo service mysql stop
```

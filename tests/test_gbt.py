import os
import numpy
import pandas
from mlflow.utils.file_utils import TempDir
from mlflow.projects import run
from mlflow import tracking
from mlflow.sklearn import load_pyfunc


def test_gbt():
    old_uri = tracking.get_tracking_uri()
    with TempDir(chdr=False, remove_on_exit=True) as tmp:
        try:
            diamonds = tmp.path("diamonds")
            artifacts = tmp.path("artifacts")
            os.mkdir(diamonds)
            os.mkdir(artifacts)
            tracking.set_tracking_uri(artifacts)
            # Download the diamonds dataset via mlflow run
            run(".", entry_point="download-example-data", version=None, 
            parameters={"dest-dir":diamonds}, experiment_id=tracking._get_experiment_id(), 
            mode="local", cluster_spec=None, git_username=None, git_password=None, use_conda=True,
            use_temp_cwd=False, storage_dir=None)

            # Keeping track of previous experiment so we can identify the new experiment's ID
            initial = os.path.join(artifacts, os.listdir(artifacts)[0])
            dir_list = os.listdir(initial)

            # Run the main dnn app via mlflow
            run(".", entry_point="gbt-regression-main", version=None, 
            parameters={"training-data-path": os.path.join(diamonds, "train_diamonds.parquet"),
                        "test-data-path": os.path.join(diamonds, "test_diamonds.parquet"), 
                        "n-trees": 100,
                        "m-depth": 3,
                        "learning-rate": .1,
                        "loss": "rmse",
                        "label-col":"price"}, 
            experiment_id=tracking._get_experiment_id(), mode="local", 
            cluster_spec=None, git_username=None, git_password=None, use_conda=True,
            use_temp_cwd=False, storage_dir=None)

            # Identifying the new experiment folder
            main = None
            for item in os.listdir(initial):
                if item not in dir_list:
                    main = item

            pyfunc = load_pyfunc(os.path.join(initial, main, "artifacts/model/model.pkl"))
            df = pandas.read_parquet(os.path.join(diamonds, "test_diamonds.parquet"))

            # Removing the price column from the DataFrame so we can use the features to predict
            df = df.drop(columns="price")

            # Predicting from the saved pyfunc
            predict = pyfunc.predict(df)

            # Make sure the data is of the right type
            assert isinstance(predict[0], numpy.float32)
        finally:
            tracking.set_tracking_uri(old_uri)

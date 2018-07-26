import os
import numpy
import pandas
from mlflow.utils.file_utils import TempDir
from mlflow.projects import run
from mlflow import tracking, tensorflow


def test_dnn():
    old_uri = tracking.get_tracking_uri()
    try:
        with TempDir(chdr=False, remove_on_exit=True) as tmp:
            diamonds = tmp.path("diamonds")
            estimator = tmp.path("estimator")
            artifacts = tmp.path("artifacts")
            os.mkdir(diamonds)
            os.mkdir(estimator)
            os.mkdir(artifacts)
            tracking.set_tracking_uri(artifacts)
            # Download the diamonds dataset via mlflow run
            run(".", entry_point="download-example-data", version=None, 
            parameters={"dest-dir":diamonds}, experiment_id=tracking._get_experiment_id(), 
            mode="local", cluster_spec=None, git_username=None, git_password=None, use_conda=True,
            use_temp_cwd=False, storage_dir=None)

            # Run the main dnn app via mlflow
            run("examples/dnn-regression", entry_point="main", version=None, 
            parameters={"model-dir": estimator,
                        "training-data-path": os.path.join(diamonds, "train_diamonds.parquet"),
                        "test-data-path": os.path.join(diamonds, "test_diamonds.parquet"), 
                        "hidden-units": "30,30", 
                        "label-col": "price", 
                        "steps":5000, 
                        "batch-size":128}, 
            experiment_id=tracking._get_experiment_id(), mode="local", 
            cluster_spec=None, git_username=None, git_password=None, use_conda=True,
            use_temp_cwd=False, storage_dir=None)

            # Loading the saved model as a pyfunc.
            pyfunc = tensorflow.load_pyfunc(os.path.join(estimator, os.listdir(estimator)[0]))

            df = pandas.read_parquet(os.path.join(diamonds, "test_diamonds.parquet"))

            predict_df = pyfunc.predict(df)
            assert 'predictions' in predict_df
            assert isinstance(predict_df['predictions'][0][0], numpy.float32)
    finally:
        tracking.set_tracking_uri(old_uri)

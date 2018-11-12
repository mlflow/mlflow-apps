# Copyright 2018 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy
import pandas
import mlflow
from mlflow.utils.file_utils import TempDir
from mlflow.projects import run
from mlflow import tracking
from mlflow.pyfunc import load_pyfunc


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
            mlflow.set_experiment("test-experiment")
            # Download the diamonds dataset via mlflow run
            run(".", entry_point="main", version=None,
                parameters={"dest-dir": diamonds},
                mode="local", cluster_spec=None, git_username=None, git_password=None,
                use_conda=True, storage_dir=None)

            # Run the main dnn app via mlflow
            submitted_run = run(
                "apps/dnn-regression", entry_point="main", version=None,
                parameters={"model-dir": estimator,
                            "train": os.path.join(diamonds, "train_diamonds.parquet"),
                            "test": os.path.join(diamonds, "test_diamonds.parquet"),
                            "hidden-units": "30,30",
                            "label-col": "price",
                            "steps": 5000,
                            "batch-size": 128},
                mode="local",
                cluster_spec=None, git_username=None, git_password=None, use_conda=True,
                storage_dir=None)

            # Loading the saved model as a pyfunc.
            pyfunc = load_pyfunc("model", submitted_run.run_id)

            df = pandas.read_parquet(os.path.join(diamonds, "test_diamonds.parquet"))

            predict_df = pyfunc.predict(df)
            assert isinstance(predict_df['predictions'][0], numpy.float32)
    finally:
        tracking.set_tracking_uri(old_uri)

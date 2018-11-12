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


def test_gbt():
    old_uri = tracking.get_tracking_uri()
    with TempDir(chdr=False, remove_on_exit=True) as tmp:
        try:
            diamonds = tmp.path("diamonds")
            artifacts = tmp.path("artifacts")
            os.mkdir(diamonds)
            os.mkdir(artifacts)
            tracking.set_tracking_uri(artifacts)
            mlflow.set_experiment("test-experiment")
            # Download the diamonds dataset via mlflow run
            run(".", entry_point="main", version=None,
                parameters={"dest-dir": diamonds},
                mode="local", cluster_spec=None, git_username=None, git_password=None,
                use_conda=True, storage_dir=None)

            # Run the main gbt app via mlflow
            submitted_run = run(
                "apps/gbt-regression", entry_point="main", version=None,
                parameters={"train": os.path.join(diamonds, "train_diamonds.parquet"),
                            "test": os.path.join(diamonds, "test_diamonds.parquet"),
                            "n-trees": 10,
                            "m-depth": 3,
                            "learning-rate": .1,
                            "loss": "rmse",
                            "label-col": "price"},
                mode="local",
                cluster_spec=None, git_username=None, git_password=None, use_conda=True,
                storage_dir=None)

            pyfunc = load_pyfunc("model", run_id=submitted_run.run_id)
            df = pandas.read_parquet(os.path.join(diamonds, "test_diamonds.parquet"))

            # Removing the price column from the DataFrame so we can use the features to predict
            df = df.drop(columns="price")

            # Predicting from the saved pyfunc
            predict = pyfunc.predict(df)

            # Make sure the data is of the right type
            assert isinstance(predict[0], numpy.float32)
        finally:
            tracking.set_tracking_uri(old_uri)

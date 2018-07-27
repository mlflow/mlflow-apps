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

import xgboost as xgb
import mlflow
from mlflow import sklearn


def train(training_pandasData, test_pandasData, label_col, feat_cols, n_trees, m_depth, 
          learning_rate, loss, training_data_path, test_data_path):

    print("training-data-path:    " + training_data_path)
    print("test-data-path:        " + test_data_path)
    print("n_trees:              ", n_trees)
    print("m-depth:              ", m_depth)
    print("learning-rate:        ", learning_rate)
    print("loss:                  " + loss)
    print("label-col:             " + label_col)
    for feat in feat_cols:
        print("feat-cols:             " + feat)

    # Split data into training labels and testing labels.
    trainingLabels = training_pandasData[label_col]
    trainingFeatures = training_pandasData[feat_cols]

    testLabels = test_pandasData[label_col]
    testFeatures = test_pandasData[feat_cols]
    
    # We will use a GBT regressor model.
    xgbr = xgb.XGBRegressor(max_depth=m_depth, 
                            learning_rate=learning_rate, 
                            n_estimators=n_trees)

    # Here we train the model
    xgbr.fit(trainingFeatures, trainingLabels, eval_metric=loss)

    # Calculating the score of the model.
    r2_score_training = xgbr.score(trainingFeatures, trainingLabels)
    r2_score_test = xgbr.score(testFeatures, testLabels)
    print("Training set score:", r2_score_training)
    print("Test set score:", r2_score_test)

    # Logging the r2 score for both sets.
    mlflow.log_metric("R2 score for training set", r2_score_training)
    mlflow.log_metric("R2 score for test set", r2_score_test)

    # Saving the model as an artifact.
    sklearn.log_model(xgbr, "model")

    run_id = mlflow.tracking.active_run().info.run_uuid
    print("Run with id %s finished" % run_id)

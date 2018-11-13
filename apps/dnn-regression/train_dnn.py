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

from sklearn.metrics import r2_score
import tensorflow as tf
import mlflow
from mlflow import tensorflow


def train(model_dir, training_pandas_data, test_pandas_data, label_col, feat_cols, hidden_units, 
          steps, batch_size, training_data_path, test_data_path):

    print("train:                 " + training_data_path)
    print("test:                  " + test_data_path)
    for hu in hidden_units:
        print("hidden-units:         ", hu)
    print("steps:                ", steps)
    print("batch_size:           ", batch_size)
    print("label-col:             " + label_col)
    for feat in feat_cols:
        print("feat-cols:             " + feat)

    # Split data into training labels and testing labels.
    training_labels = training_pandas_data[label_col].values
    test_labels = test_pandas_data[label_col].values

    training_features = {}  # Dictionary of column names to column values for training
    test_features = {}  # Dictionary of column names to column values for testing
    tf_feat_cols = []  # List of tf.feature_columns for input functions
    feature_spec = {}  # Dictionary of column name -> placeholder tensor for receiver functions
    # Create TensorFlow columns based on passed in feature columns
    for feat in feat_cols:
        training_features[feat] = training_pandas_data[feat].values
        test_features[feat] = test_pandas_data[feat].values
        tf_feat_cols.append(tf.feature_column.numeric_column(feat))
        feature_spec[feat] = tf.placeholder("float", name=feat, shape=[None])

    # Create receiver function for loading the model for serving.
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Create input functions for both the training and testing sets.
    input_train = tf.estimator.inputs.numpy_input_fn(training_features, training_labels, 
                                                     shuffle=True, batch_size=batch_size)
    input_test = tf.estimator.inputs.numpy_input_fn(test_features, test_labels, 
                                                    shuffle=False, batch_size=batch_size)

    # Creating DNNRegressor
    regressor = tf.estimator.DNNRegressor(
        feature_columns=tf_feat_cols,
        hidden_units=hidden_units,
        model_dir=model_dir)

    # Training regressor on training input function
    regressor.train(
        input_fn=input_train,
        steps=steps)

    # Evaluating model on training data
    test_eval = regressor.evaluate(input_fn=input_test)

    # Calculating the RMSE of the testing set.
    test_rmse = test_eval["average_loss"]**0.5

    # Putting the predictions in a list we can use to calculate r^2 scores with.
    train_raw_pred = list(regressor.predict(input_fn=input_train))
    train_pred = [i['predictions'][0] for i in train_raw_pred]
    r2_train = r2_score(training_labels, train_pred)

    test_raw_pred = list(regressor.predict(input_fn=input_test))
    test_pred = [i['predictions'][0] for i in test_raw_pred]
    r2_test = r2_score(test_labels, test_pred)

    print("Test RMSE:", test_rmse)
    print("Training Set R2", r2_train)
    print("Test Set R2", r2_test)

    mlflow.log_param("num_train_points", len(training_pandas_data[label_col].values))

    #Logging the RMSE and r2 scores.
    mlflow.log_metric("Test RMSE", test_rmse)
    mlflow.log_metric("Train R2", r2_train)
    mlflow.log_metric("Test R2", r2_test)

    # Saving TensorFlow model.
    saved_estimator_path = regressor.export_savedmodel(model_dir, 
                                                       receiver_fn).decode("utf-8")

    # Logging the TensorFlow model just saved.
    tensorflow.log_model(
        tf_saved_model_dir=saved_estimator_path,
        tf_meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
        tf_signature_def_key="predict",
        artifact_path="model")

    run_id = mlflow.active_run().info.run_uuid
    print("Run with id %s finished" % run_id)

import os
import pickle
import mlflow
from mlflow import log_metric, log_param, tensorflow
import tensorflow as tf

# mlflow run mlflow-examples -e dnn-regression-main 
# -P model-dir="mlflow-examples/dnn-regression/estimator" 
# -P training-data-path="mlflow-examples/diamonds/train_diamonds.parquet" 
# -P test-data-path="mlflow-examples/diamonds/test_diamonds.parquet" 
# -P hidden-units="30,30" -P label-col="price" -P steps=5000 -P batch-size=128

def train(model_dir, training_pandasData, test_pandasData, label_col, feat_cols, hidden_units, 
          steps, batch_size, training_data_path, test_data_path):

    print("training-data-path:    " + training_data_path)
    print("test-data-path:        " + test_data_path)
    for hu in hidden_units:
        print("hidden-units:         ", hu)
    print("steps:                ", steps)
    print("batch_size:           ", batch_size)
    print("label-col:             " + label_col)
    for feat in feat_cols:
        print("feat-cols:             " + feat)

    # Split data into a labels dataframe and a features dataframe
    trainingLabels = training_pandasData[label_col].values
    testLabels = test_pandasData[label_col].values

    trainingFeatures = {} # Dictionary of column names to column values for training
    testFeatures = {} # Dictionary of column names to column values for testing
    tf_feat_cols = [] # List of tf.feature_columns for input functions
    feature_spec = {} # Dictionary of column name -> placeholder tensor for receiver functions
    # Create TensorFlow columns based on passed in feature columns
    for feat in feat_cols:
        trainingFeatures[feat] = training_pandasData[feat].values
        testFeatures[feat] = test_pandasData[feat].values
        tf_feat_cols.append(tf.feature_column.numeric_column(feat))
        feature_spec[feat] = tf.placeholder("double", name=feat, shape=[None])

    # Create receiver function for loading the model for serving.
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Create input functions for both the training and testing sets.
    input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures, 
                                                    trainingLabels, 
                                                    shuffle=True, 
                                                    batch_size=batch_size)
    input_test = tf.estimator.inputs.numpy_input_fn(testFeatures, 
                                                    testLabels, 
                                                    shuffle=False, 
                                                    batch_size=batch_size)
    
    # Creating DNNRegressor
    regressor = tf.estimator.DNNRegressor(
        feature_columns=tf_feat_cols,
        hidden_units=hidden_units)

    # Training regressor on training input function
    regressor.train(
        input_fn=input_train,
        steps=steps)

    # Evaluating model on training data
    test_eval = regressor.evaluate(input_fn=input_test)

    test_rmse = test_eval["average_loss"]**0.5

    print("Test RMSE:", test_rmse)

    log_param("Number of data points", len(training_pandasData[label_col].values))

    #Logging the RMSE and predictions.
    log_metric("RMSE for test set", test_rmse)
    
    # Saving TensorFlow model.
    saved_estimator_path = regressor.export_savedmodel(model_dir, 
                                                       receiver_fn).decode("utf-8")

    # Logging the TensorFlow model just saved.
    tensorflow.log_saved_model(saved_model_dir=saved_estimator_path,
                                      signature_def_key="predict", 
                                      artifact_path="model")

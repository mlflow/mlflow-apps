import pandas
import tensorflow as tf
from mlflow import log_metric, log_parameter, log_output_files, active_run_id
from mlflow.sklearn import log_model

#python dnn-regression/main_dnn.py "/Users/juntaizheng/mlflow-examples/airbnb_train.parquet" "10" 10 10 price

def train(training_pandasData, test_pandasData, label_col, feat_cols, hidden_units, steps, batch_size, training_data_path, test_data_path):

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

    trainingFeatures = {}
    testFeatures = {}
    for feat in feat_cols:
        trainingFeatures[feat.replace(" ", "_")] = training_pandasData[feat].values
        testFeatures[feat.replace(" ", "_")] = test_pandasData[feat].values

    # Create input functions for both the training and testing sets.
    with tf.Session() as session:
        input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures, trainingLabels, shuffle=True, batch_size=batch_size)
        input_test = tf.estimator.inputs.numpy_input_fn(testFeatures, testLabels, shuffle=False, batch_size=batch_size)
    
    # Create TensorFlow columns based on passed in feature columns
    tf_feat_cols = []
    for col in feat_cols:
        tf_feat_cols.append(tf.feature_column.numeric_column(col.replace(" ", "_")))

    # Creating DNNRegressor
    regressor = tf.estimator.DNNRegressor(
        feature_columns=tf_feat_cols,
        hidden_units=hidden_units)

    # Training regressor on training input function
    regressor.train(
        input_fn=input_train,
        steps=steps)

    # Evaluating model on training data
    training_eval = regressor.evaluate(input_fn=input_train)
    test_eval = regressor.evaluate(input_fn=input_test)

    training_rmse = training_eval["average_loss"]**0.5
    test_rmse = test_eval["average_loss"]**0.5

    print("Training RMSE:", training_rmse)
    print("Test RMSE:", test_rmse)

    print("Logging parameters.")
    log_parameter("Training data path", training_data_path)
    log_parameter("Test data path", test_data_path)
    log_parameter("Label column", label_col)
    log_parameter("Feature columns", feat_cols)
    log_parameter("Hidden units", hidden_units)
    log_parameter("Steps", steps)
    log_parameter("Batch size", batch_size)
    log_parameter("Number of data points", len(training_pandasData[label_col].values))

    #Logging the rmse for both sets.
    log_metric("RMSE for training set", training_rmse)
    log_metric("RMSE for test set", test_rmse)

    log_output_files("outputs")

    #TODO: Saving the model as an artifact.
    #direct = regressor.export_savedmodel("../mlruns", tf.estimator.export.build_parsing_serving_input_receiver_fn(trainingFeatures))

    #print("Model saved in mlruns/%s" % direct)

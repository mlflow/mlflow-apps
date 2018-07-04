import xgboost as xgb
from mlflow import log_metric, sklearn


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
    log_metric("R2 score for training set", r2_score_training)
    log_metric("R2 score for test set", r2_score_test)

    # Saving the model as an artifact.
    sklearn.log_model(xgbr, "model")

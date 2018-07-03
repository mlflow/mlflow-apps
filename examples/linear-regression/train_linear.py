from sklearn.linear_model import ElasticNet
from mlflow import log_metric, sklearn


def train(training_pandas_data, test_pandas_data, label_col, 
          feat_cols, alpha, l1_ratio, training_data_path, test_data_path):

    print("training-data-path:    " + training_data_path)
    print("test-data-path:        " + test_data_path)
    print("alpha:        ", alpha)
    print("l1-ratio:     ", l1_ratio)
    print("label-col:     " + label_col)
    for col in feat_cols:
        print("feat-cols:     " + col)

    # Split data into training labels and testing labels.
    trainingLabels = training_pandas_data[label_col].values
    trainingFeatures = training_pandas_data[feat_cols].values

    testLabels = test_pandas_data[label_col].values
    testFeatures = test_pandas_data[feat_cols].values

    #We will use a linear Elastic Net model.
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    # Here we train the model and keep track of how long it takes.
    en.fit(trainingFeatures, trainingLabels)

    # Calculating the score of the model.
    r2_score_training = en.score(trainingFeatures, trainingLabels)
    r2_score_test = 0
    r2_score_test = en.score(testFeatures, testLabels)
    print("Training set score:", r2_score_training)
    print("Test set score:", r2_score_test)

    #Logging the r2 score for both sets.
    log_metric("R2 score for training set", r2_score_training)
    log_metric("R2 score for test set", r2_score_test)

    #Saving the model as an artifact.
    sklearn.log_model(en, "model")

from time import time
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import train_test_split
from mlflow import log_metric, log_parameter, log_output_files, active_run_id
from mlflow.sklearn import log_model

def train(pandasData, label_col, feat_cols, test_percent, alpha, l1_ratio, data_path):
    if data_path:
        print("data-path:    ", data_path)
    print("alpha:        ", alpha)
    print("l1-ratio:     ", l1_ratio)
    print("test-percent: ", test_percent)
    print("label-col:     " + label_col)
    for col in feat_cols:
        print("feat-cols:     " + col)

    # Split data into a labels dataframe and a features dataframe
    labels = pandasData[label_col].values
    features = pandasData[feat_cols].values

    # Hold out test_percent of the data for testing.  We will use the rest for training.
    trainingFeatures, testFeatures, trainingLabels, testLabels = train_test_split(features, 
                                                                labels, test_size=test_percent)
    ntrain, ntest = len(trainingLabels), len(testLabels)
    print("Split data randomly into {} training and {} test instances.".format(ntrain, ntest))

    #We will use a linear Elastic Net model.
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    # Here we train the model and keep track of how long it takes.
    start_time = time()
    en.fit(trainingFeatures, trainingLabels)

    # Calculating the score of the model.
    r2_score_training = en.score(trainingFeatures, trainingLabels)
    r2_score_test = 0
    if test_percent != 0:
        r2_score_test = en.score(testFeatures, testLabels)
    timed = time() - start_time
    print("Training set score:", r2_score_training)
    if test_percent != 0:
        print("Test set score:", r2_score_test)

    #Logging the parameters for viewing later. Can be found in the folder mlruns/.
    if data_path:
        log_parameter("Data Path", data_path)
    log_parameter("Alpha", alpha)
    log_parameter("l1 ratio", l1_ratio)
    log_parameter("Testing set percentage", test_percent)
    log_parameter("Label column", label_col)
    log_parameter("Feature columns", feat_cols)
    log_parameter("Number of data points", len(features))

    #Logging the r2 score for both sets.
    log_metric("R2 score for training set", r2_score_training)
    if test_percent != 0:
        log_metric("R2 score for test set", r2_score_test)

    log_output_files("outputs")

    #Saving the model as an artifact.
    log_model(en, "model")

    print("Model saved in mlruns/%s" % active_run_id())

    #Determining how long the program took.
    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)

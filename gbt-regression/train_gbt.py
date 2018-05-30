import pandas
import xgboost as xgb

from mlflow import log_metric, log_parameter, log_output_files, active_run_id
from mlflow.sklearn import log_model, save_model

from sklearn import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
from sklearn.metrics import *

from time import time

def train(args, pandasData):

	# Split data into a labels dataframe and a features dataframe
	labels = pandasData[args.label_col].values
	features = pandasData[args.feat_cols].values

	# Hold out test_percent of the data for testing.  We will use the rest for training.
	trainingFeatures, testFeatures, trainingLabels, testLabels = train_test_split(features, labels, test_size=args.test_percent)
	ntrain, ntest = len(trainingLabels), len(testLabels)
	print("Split data randomly into 2 sets: {} training and {} test instances.".format(ntrain, ntest))

	# We will use a GBT regressor model.
	xgbr = xgb.XGBRegressor(max_depth = args.m_depth, learning_rate = args.learning_rate, n_estimators = args.n_trees)

	# Here we train the model and keep track of how long it takes.
	start_time = time()
	xgbr.fit(trainingFeatures, trainingLabels, eval_metric = args.loss)

	# Calculating the score of the model.
	r2_score_training = xgbr.score(trainingFeatures, trainingLabels)
	r2_score_test = 0
	if args.test_percent != 0:
		r2_score_test = xgbr.score(testFeatures, testLabels)
	timed = time() - start_time
	print("Training set score:", r2_score_training)
	if args.test_percent != 0:
		print("Test set score:", r2_score_test)

	#Logging the parameters for viewing later. Can be found in the folder mlruns/.
	if len(vars(args)) > 7:
		log_parameter("Data Path", args.data_path)
	log_parameter("Number of trees", args.n_trees)
	log_parameter("Max depth of trees", args.m_depth)
	log_parameter("Learning rate", args.learning_rate)
	log_parameter("Testing set percentage", args.test_percent)
	log_parameter("Loss function used", args.loss)
	log_parameter("Label column", args.label_col)
	log_parameter("Feature columns", args.feat_cols)
	log_parameter("Number of data points", len(features))

	#Logging the r2 score for both sets.
	log_metric("R2 score for training set", r2_score_training)
	if args.test_percent != 0:
		log_metric("R2 score for test set", r2_score_test)

	log_output_files("outputs")

	#Saving the model as an artifact.
	log_model(xgbr, "model")

	print("Model saved in mlruns/%s" % active_run_id())

	#Determining how long the program took.
	print("This model took", timed, "seconds to train and test.")
	log_metric("Time to run", timed)

import argparse
import pandas
import train_dnn
import utils

# Trains a single-machine Tensorflow DNNRegressor model on the provided data file, 
# producing a pickled model file. Uses MLflow tracking APIs to log the input parameters, 
# the model file, and the model's training loss.

#Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("training_data_path", help="Path to training parquet dataset file.",
                    type=str)
parser.add_argument("test_data_path", help="Path to test parquet dataset file.",
                    type=str)
parser.add_argument("hidden_units", help="Hidden layer dimensions for the model.",
                    type=lambda s: [str(i) for i in s.split(',')])
parser.add_argument("steps", help="Number of steps for the training of the model to take.",
                    type=int)
parser.add_argument("batch_size", help="Number of steps for the training of the model to take.",
                    type=int)
parser.add_argument("label_col", help="Name of label column.",
                    type=str)
parser.add_argument("--feat-cols", help="List of feature column names. "
                        "Input must be a single string with columns delimited by commas.",
                    type=lambda s: [str(i) for i in s.split(',')])

args = parser.parse_args()

# Reading the parquet file into a pandas dataframe.
training_pandasData = pandas.read_parquet(args.training_data_path)

test_pandasData = pandas.read_parquet(args.test_data_path)

# Handle determining feature columns.
feat_cols = utils.get_feature_cols(args.feat_cols, args.label_col, list(training_pandasData))

# Concert hidden units from string to int list.
hidden_units = []
for hu in args.hidden_units:
    hidden_units.append(int(hu))

# Train the model based on the parameters provided.
train_dnn.train(training_pandasData, test_pandasData, args.label_col, feat_cols, 
                    hidden_units, args.steps, args.batch_size, 
                    args.training_data_path, args.test_data_path)

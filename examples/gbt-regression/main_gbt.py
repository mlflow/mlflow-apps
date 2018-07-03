import argparse
import pandas
import train_gbt
import utils

# Trains a single-machine scikit-learn GBT model on the provided data file, 
# producing a pickled model file. Uses MLflow tracking APIs to log the input parameters, 
# the model file, and the model's training loss.

# Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("training_data_path", help="Path to training parquet dataset file.", type=str)
parser.add_argument("test_data_path", help="Path to test parquet dataset file.", type=str)
parser.add_argument("n_trees", help="Number of trees to fit.", type=int)
parser.add_argument("m_depth", help="Max depth of trees.", type=int)
parser.add_argument("learning_rate", help="Learning rate of the model.", type=float)
parser.add_argument("loss", help="""Loss function to use. See 
                    https://github.com/dmlc/xgboost/blob/master/doc/parameter.md 
                    for list of functions.""", type=str)
parser.add_argument("label_col", help="Name of label column.", type=str)
parser.add_argument("--feat-cols", help="""List of feature column names. "
                    Input must be a single string with columns delimited by commas.""",
                    type=lambda s: [str(i) for i in s.split(',')])
args = parser.parse_args()

# Reading the parquet file into a pandas dataframe.
training_pandasData = pandas.read_parquet(args.training_data_path)

test_pandasData = pandas.read_parquet(args.test_data_path)

# Handle determining feature columns.
feat_cols = utils.get_feature_cols(args.feat_cols, args.label_col, list(training_pandasData))

# Train the model based on the parameters provided.
train_gbt.train(training_pandasData, test_pandasData, args.label_col, feat_cols, 
                    args.n_trees, args.m_depth, args.learning_rate, args.loss, 
                    args.training_data_path, args.test_data_path)

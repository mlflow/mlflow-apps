import argparse
from shutil import rmtree
from tempfile import mkdtemp
import train_linear
import utils

# Trains a single-machine scikit-learn Elastic Net model on the provided data file, 
# producing a pickled model file. Uses MLflow tracking APIs to log the input parameters,
# the model file, and the model's training loss.

#Parsing arguments.
parser = argparse.ArgumentParser()

parser.add_argument("alpha", help="Alpha value for Elastic Net linear regressor.",
                    type=float)
parser.add_argument("l1_ratio", help="L1 ratio for Elastic Net linear regressor. "
    "See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html "
    "for more details.",
                    type=float)
parser.add_argument("test_percent", help="Percent of data to use as test data.",
                    type=float)
parser.add_argument("label_col", help="Name of label column.",
                    type=str)
parser.add_argument("--feat-cols", help="List of feature column names. "
                        "Input must be a single string with columns delimited by commas.",
                    type=lambda s: [str(i) for i in s.split(',')])

args = parser.parse_args()

# Creating a temporary directory for the storage of the csv and parquet file. 
# Will be deleted at the end of the script.
temp_folder_path = mkdtemp()

pandasData = utils.download_diamonds(temp_folder_path)

feat_cols = utils.get_feature_cols(args.feat_cols, args.label_col, list(pandasData))

# Train the model based on the parameters provided.
try:
    train_linear.train(pandasData, args.label_col, feat_cols, 
                        args.test_percent, args.alpha, args.l1_ratio, None)

# Delete the temporary folder that stores the csv and parquet files.
finally:
    rmtree(temp_folder_path)

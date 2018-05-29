import argparse
import pandas
import train_gbt

"""
Trains a single-machine scikit-learn GBT model on the provided data file, producing a pickled model file. 
Uses MLflow tracking APIs to log the input parameters, the model file, and the model's training loss.
"""

# Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Path to parquet dataset file.",
                    type=str)
parser.add_argument("n_trees", help="Number of trees to fit.",
                    type=int)
parser.add_argument("m_depth", help="Max depth of trees.",
                    type=int)
parser.add_argument("learning_rate", help="Learning rate of the model.",
					type=float)
parser.add_argument("test_percent", help="Percent of data to use as test data.",
					type=float)
parser.add_argument("loss", help="""Loss function to use. See 
								https://github.com/dmlc/xgboost/blob/master/doc/parameter.md for list of functions.""",
                    type=str)
parser.add_argument("label_col", help="Name of label column.",
                    type=str)
parser.add_argument("feat_cols", help="List of feature column names. Input must be a single string with columns delimited by commas.",
                    type=lambda s: [str(i) for i in s.split(',')])

args = parser.parse_args()

print("data-path:    ", args.data_path)
print("n-trees:      ", args.n_trees)
print("m-depth:      ", args.m_depth)
print("learning-rate:", args.learning_rate)
print("test-percent: ", args.test_percent)
print("loss:          " + args.loss)
print("label-col:     " + args.label_col)
for i in args.feat_cols:
	print("feat-cols      " + i)

#Reading the parquet file into a pandas dataframe.
pandasData = pandas.read_parquet(args.data_path)

# Train the model based on the parameters provided.
train_gbt.train(args, pandasData)

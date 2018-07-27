# Copyright 2018 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pandas
import train_dnn
import utils

# Trains a single-machine Tensorflow DNNRegressor model on the provided data file, 
# producing a pickled model file. Uses MLflow tracking APIs to log the input parameters, 
# the model file, and the model's training loss. It is assumed that the columns input are numeric.

#Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Directory to save the model in.", type=str)
parser.add_argument("training_data_path", help="Path to training parquet dataset file.", type=str)
parser.add_argument("test_data_path", help="Path to test parquet dataset file.", type=str)
parser.add_argument("hidden_units", help="Hidden layer dimensions for the model.",
                    type=lambda s: [str(i) for i in s.split(',')])
parser.add_argument("steps", help="Number of steps for training the model will take.", type=int)
parser.add_argument("batch_size", help="Size of batches for model.", type=int)
parser.add_argument("label_col", help="Name of label column.", type=str)
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
train_dnn.train(args.model_dir, training_pandasData, test_pandasData, args.label_col, 
                feat_cols, hidden_units, args.steps, args.batch_size, 
                args.training_data_path, args.test_data_path)

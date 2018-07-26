# dnn-regression

This sample project creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. The arguments to the program are as follows:
* `model-dir`: (str, required) local path or URI of a directory to which the DNNRegressor's checkpoints and the final exported SavedModel will be written.
* `training-data-path`: (str, required) local path or URI of a parquet file containing training data
* `test-data-path`: (str, required) local path or URI of a parquet file containing test data
* `hidden-units`: (str, required) size and number of layers for the dnn; `string` input which has layers delimited by commas (i.e. "10,10" for two layers of 10 nodes each)
* `steps`: (int) steps to be run while training the regressor; default `100`
* `batch-size`: (int) batch size used for creation of input functions for training and evaluation; default `128`
* `label-col`: (str, required) name of label column in dataset
* `feat-cols`: (str) names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical. Column names must adhere to TensorFlow [constraints](https://www.tensorflow.org/api_docs/python/tf/Operation#__init__).

To run the app with default parameters while in the root directory, run the command 
```
mlflow run examples/dnn-regression/ -e main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```
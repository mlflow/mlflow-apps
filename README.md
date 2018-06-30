# mlflow-examples
## [MLflow](http://mlflow.org) App Library
### gbt-regression
This app creates and fits a Gradient Boosted Tree model based on parquet-formatted input data. The arguments to the program are as follows:
* `data-path`: local or absolute path to the parquet file to be fitted on; `string` input
* `n-trees`: number of trees for the regressor; default `100`
* `m-depth`: maximum depth of trees of regressor; default `10`
* `learning-rate`: learning rate of the model; ranges from `0.0` to `1.0`; default `.2`
* `test-percent`: percentage of the input data that is held as the testing set; ranges from `0.0-1.0`; default `.3`
* `loss`: name of [loss function](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) to be used; default `"rmse"`
* `label-col`: name of label column in dataset; `string` input
* `feat-cols`: names of columns in dataset to be used as features; input is one `string` with names delimited by commas

This app currently assumes that the input data is all numerical.

The following commands should be run from the root repository directory:

To run the app with default parameters on a dataset, run the code 
```
mlflow run . -e gbt-regression-main -P data-path="insert/data/path/" -P label-col="insert.label.col" -P feat-cols="insert,feat,cols"
```
where `insert/data/path/` is replaced with the actual path to the parquet data, `insert.label.col` is replaced with the label column, and `insert,feat,cols` is replaced with a comma delimited string of feature column names.

To run an example of the app on the [diamonds dataset](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv), run the code 
```
mlflow run . -e gbt-regression-example -P label-col="price" -P feat-cols="carat,cut,color,clarity,depth,table,x,y,z"
```

To pass parameter values to the app, simply add `-P name-of-argument=value.of.argument` to the command. An example of adding custom parameters on the diamonds dataset is as follows: 
```
mlflow run . -e gbt-regression-example -P n-trees=50 -P m-depth=20 -P learning-rate=.4 -P test-percent=.1 -P label-col="price" -P feat-cols="carat,cut,color,clarity,depth,table,x,y,z"
```

To run an app from a different directory, replace the "." with the path to the root repository folder. For example, the command to run the app on the diamonds dataset from the parent directory of `mlflow-examples` is:
```
mlflow run mlflow-examples -e gbt-regression-example -P label-col="price" -P feat-cols="carat,cut,color,clarity,depth,table,x,y,z"
```

### linear-regression

This app creates and fits a [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) model based on parquet-formatted input data. The arguments to the program are as follows:
* `training-data-path`: local or absolute path to the parquet file to be fitted on; `string` input
* `test-data-path`: local or absolute path to the parquet file to be tested on; `string` input
* `alpha`: alpha for the regressor; default `.001`
* `l1-ratio`: l1 ratio to be used for the regressor; default `.5`
* `label-col`: name of label column in dataset; `string` input
* `feat-cols`: names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

The following commands should be run from the root repository directory:

To run the app with default parameters on a dataset, run the code 
```
mlflow run . -e linear-regression-main -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P label-col="insert.label.col"
```
where `insert/data/path/` is replaced with the actual path to the parquet data, `insert.label.col` is replaced with the label column.

To run an app from a different directory, replace the "." with the path to the root repository folder. For example, the command to run the app from the parent directory of `mlflow-examples` is:
```
mlflow run mlflow-examples -e linear-regression-main -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P label-col="price" 
```

### dnn-regression

This sample project creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. The arguments to the program are as follows:
* `training-data-path`: (str) local path or URI of a parquet file containing training data
* `test-data-path`: (str) local path or URI of a parquet file containing test data
* `hidden-units`: (str) size and number of layers for the dnn; `string` input has layers delimited by commas (i.e. "10,10" for two layers of 10 nodes each)
* `steps`: (int) steps to be run whil training the regressor; default `100`
* `batch-size`: (int) batch size used for creation of input functions for training and evaluation; default `128`
* `label-col`: (str) name of label column in dataset
* `feat-cols`: (str) names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This example code currently only works for numerical data. In addition, column names must adhere to TensorFlow [constraints](https://www.tensorflow.org/api_docs/python/tf/Operation#__init__).

To run the project locally with default parameters on a dataset while in the parent directory, run the command
```
mlflow run . -e dnn-regression-main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```
where `insert/model/save/path` is the directory to which the DNNRegressor's checkpoints and the final exported SavedModel will be written.

You can download example training & test parquet files containing the [diamonds](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv) dataset by running 
```
mlflow run  . -e download-example-data -P dest_dir="path/to/dir"
```
You can then use these files as data for running the example application.

To run the project from a git repository, run the command
```
mlflow run git@github.com:databricks/mlflow-examples.git -v master -e dnn-regression-main -P model-dir="insert/model/save/path" -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```

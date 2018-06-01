# mlflow-examples
## MLflow App Library
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

This app creates and fits a Gradient Boosted Tree model based on parquet-formatted input data. The arguments to the program are as follows:
* `data-path`: local or absolute path to the parquet file to be fitted on; `string` input
* `alpha`: alpha for the regressor; default `.001`
* `l1-ratio`: l1 ratio to be used for the regressor; default `.5`
* `test-percent`: percentage of the input data that is held as the testing set; ranges from `0.0-1.0`; default `.3`
* `label-col`: name of label column in dataset; `string` input
* `feat-cols`: names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

The following commands should be run from the root repository directory:

To run the app with default parameters on a dataset, run the code 
```
mlflow run . -e linear-regression-main -P data-path="insert/data/path/" -P label-col="insert.label.col"
```
where `insert/data/path/` is replaced with the actual path to the parquet data, `insert.label.col` is replaced with the label column.

To run an example of the app on the [diamonds dataset](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv), run the code 
```
mlflow run . -e linear-regression-example -P label-col="price"
```

To pass parameter values to the app, simply add `-P name-of-argument=value.of.argument` to the command. An example of passing custom parameters using the diamonds dataset is as follows: 
```
mlflow run . -e linear-regression-example -P alpha=.01 -P l1-ratio=.2 -P test-percent=.1 -P label-col="price" -P feat-cols="carat,cut,color,clarity,depth,table,x,y,z"
```

To run an app from a different directory, replace the "." with the path to the root repository folder. For example, the command to run the app on the diamonds dataset from the parent directory of `mlflow-examples` is:
```
mlflow run mlflow-examples -e linear-regression-example -P label-col="price" 
```

### deep neural network regression

This app creates and fits a Tensorflow DNNRegressor model based on parquet-formatted input data. The arguments to the program are as follows:
* `training-data-path`: local or absolute path to the parquet file to be fitted on; `string` input
* `test-data-path`: local or absolute path to the parquet file to be tested on; `string` input
* `hidden-units`: size and number of layers for the dnn; `string` input with layers delimited by commas (i.e. "10,10")
* `steps`: steps to be run whil training the regressor; default `100`
* `batch-size`: batch size used for creation of input functions for training and evaluation; default `128`
* `label-col`: name of label column in dataset; `string` input
* `feat-cols`: names of columns in dataset to be used as features; input is one `string` with names delimited by commas
    * This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

The following commands should be run from the root repository directory:

To run the app with default parameters on a dataset, run the code 
```
mlflow run . -e dnn-regression-main -P training-data-path="insert/data/path/" -P test-data-path="insert/data/path/" -P hidden-units="10,10" -P label-col="insert.label.col"
```
where `insert/data/path/` is replaced with the actual path to the parquet data, `insert.label.col` is replaced with the label column.

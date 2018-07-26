# gbt-regression
This app creates and fits an [XGBoost Gradient Boosted Tree](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) model based on parquet-formatted input data. The arguments to the program are as follows:
* `training-data-path`: (str, required) local path or URI of a parquet file containing training data
* `test-data-path`: (str, required) local path or URI of a parquet file containing test data
* `n-trees`: (int) number of trees for the regressor; default `100`
* `m-depth`: (int) maximum depth of trees of regressor; default `10`
* `learning-rate`: (float) learning rate of the model; ranges from `0.0` to `1.0`; default `.2`
* `loss`: (str) name of [loss function](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) to be used; default `"rmse"`
* `label-col`: (str, required) name of label column in dataset; `string` input
* `feat-cols`: (str) names of columns in dataset to be used as features; input is one `string` with names delimited by commas
	* This argument is optional. If no argument is provided, it is assumed that all columns but the label column are feature columns.

This app currently assumes that the input data is all numerical.

To run the app with default parameters while in the root directory, run the command 
```
mlflow run examples/gbt-regression -e main -P data-path="insert/data/path/" -P label-col="insert.label.col" 
```
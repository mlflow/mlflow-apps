# [MLflow](http://mlflow.org) App Library

Collection of pluggable MLflow apps (MLflow projects). You can call the apps in this repository to:
* Seamlessly embed ML functionality into your own applications
* Reproducibly train models from a variety of frameworks on big & small data, without worrying about installing dependencies

## Getting Started
### Running Apps via the CLI
Let's start by running the gbt-regression app, which trains an XGBoost Gradient Boosted Tree model.

First, download example training & test parquet files by running:
 
```
temp="$(mktemp -d)"
mlflow run git@github.com:databricks/mlflow-apps.git -e download-example-data -P dest-dir=$temp
```

This will download the diamonds [diamonds](https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv) dataset to the directory `temp-data`.

Then, train a GBT model on the data, saving the fitted network as an MLflow model. See the [gbt-regression docs](examples/gbt-regression/README.md) for more info on available parameters.
```
mlflow run git@github.com:databricks/mlflow-apps.git#examples/gbt-regression/ -P training-data-path="$temp/train_diamonds.parquet" -P test-data-path="$temp/test_diamonds.parquet" -P label-col="price"
```
The output will contain a line with the run ID, e.g:
```
Run with ID <run id> finished
```

We can now use the fitted model to make predictions on our test data via the MLflow CLI and the run id produced by the previous command:
```
mlflow pyfunc predict -m model -r <run id> -i "$temp/diamonds.csv"
```
The output of this command will be 20 numbers, which are predictions of 20 diamonds' prices based on their features (located in `temp-data/diamonds.csv`). You can compare these numbers to the actual prices of the diamonds, which are located in `temp-data/actual_diamonds.csv`.

### Calling an App in Your Code

Calling an app from your code is simple  - just use MLflow's [Python API](https://mlflow.org/docs/latest/projects.html#building-multi-step-workflows):
```
# Train an XGBoost GBT, exporting it as an MLflow model
train_data_path = "..."
test_data_path = "..."
label_col = "..."
# Running the MLflow project
submitted_run = mlflow.projects.run(uri="git@github.com:databricks/mlflow-apps.git#examples/gbt-regression/", parameters={"training-data-path":train_data_path, "test-data-path":test_data_path, "label-col":label_col})
# Load the model again for inference or more training
model = mlflow.sklearn.load_model("model", submitted_run.run_id)
```

## Apps

The library contains the following apps:

### dnn-regression

This app creates and fits a Tensorflow [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) model based on parquet-formatted input data. Then, the application exports the model to a local file and logs the model using MLflow's APIs. See more info [here](examples/dnn-regression/).

### gbt-regression
This app creates and fits an [XGBoost Gradient Boosted Tree](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) model based on parquet-formatted input data. See more info [here](examples/gbt-regression/).

### linear-regression

This app creates and fits an [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) model based on parquet-formatted input data. See more info [here](examples/linear-regression/).

## Contributing

If you would like to contribute to this library, please see the [contribution guide](CONTRIBUTING.md) for details.

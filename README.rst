`MLflow`_ App Library
=====================

Collection of pluggable MLflow apps (MLflow projects). You can call the
apps in this repository to:

- Seamlessly embed ML functionality into your own applications
- Reproducibly train models from a variety of frameworks on big & small data, without worrying about installing dependencies

We recommend calling the apps in this library from a Python 3 environment - the apps run in Python 3 conda environments, so it may not be possible to load the models produced by the apps back into Python 2 environments.

Getting Started
---------------

Running Apps via the CLI
~~~~~~~~~~~~~~~~~~~~~~~~

Let’s start by running the gbt-regression app, which trains an XGBoost
Gradient Boosted Tree model.

First, download example training & test parquet files containing the
`diamonds`_:

::

   temp="$(mktemp -d)"
   mlflow run https://github.com/mlflow/mlflow-apps.git -P dest-dir=$temp

Then, train a GBT model and save it as an MLflow model (see the `GBT App
docs`_ for more information):

::

   mlflow run https://github.com/mlflow/mlflow-apps.git#apps/gbt-regression/ -P train="$temp/train_diamonds.parquet" -P test="$temp/test_diamonds.parquet" -P label-col="price"

The output will contain a line with the run ID, e.g:

::

   Run with ID <run id> finished

We can now use the fitted model to predict on our test data (substitute
in the run ID from the previous step):

::

   mlflow pyfunc predict -m model -r <run id> -i "$temp/diamonds.csv"

The output of this command will be 20 numbers, which are predictions of
20 diamonds’ prices based on their features (located in
``$temp/diamonds.csv``). You can compare these numbers to the actual
prices of the diamonds, which are viewable via

::

   cat $temp/diamond_prices.csv

Finally, clean up the generated files via:

::

   rm -r $temp

Calling an App in Your Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling an app from your code is simple - just use MLflow’s `Python
API`_:

::

   # Train an XGBoost GBT, exporting it as an MLflow model
   train_data_path = "..."
   test_data_path = "..."
   label_col = "..."
   # Running the MLflow project
   submitted_run = mlflow.projects.run(uri="https://github.com/mlflow/mlflow-apps.git#apps/gbt-regression/", parameters={"train":train_data_path, "test":test_data_path, "label-col":label_col})
   # Load the model again for inference or more training
   model = mlflow.sklearn.load_model("model", submitted_run.run_id)

Apps
----

The library contains the following apps:

dnn-regression
~~~~~~~~~~~~~~

This app creates and fits a TensorFlow `DNNRegressor`_ model based on
parquet-formatted input data. Then, the application exports the model to
a local file and logs the model using MLflow’s APIs. See more info
`here`_.

gbt-regression
~~~~~~~~~~~~~~

This app creates and fits an `XGBoost Gradient Boosted Tree`_ model
based on parquet-formatted input data. See more info
`here <apps/gbt-regression/>`__.

linear-regression
~~~~~~~~~~~~~~~~~

This app creates and fits an `Elastic Net`_ model based on
parquet-formatted input data. See more info
`here <apps/linear-regression/>`__.

Contributing
------------

If you would like to contribute to this library, please see the
`contribution guide`_ for details.


.. _MLflow: http://mlflow.org
.. _diamonds: https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv
.. _GBT App docs: apps/gbt-regression/README.md
.. _Python API: https://mlflow.org/docs/latest/projects.html#building-multi-step-workflows
.. _DNNRegressor: https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor
.. _XGBoost Gradient Boosted Tree: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
.. _here: apps/dnn-regression/
.. _Elastic Net: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
.. _contribution guide: CONTRIBUTING.rst

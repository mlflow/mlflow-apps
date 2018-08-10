How to Contribute
=================

Running the Tests
-----------------
Install the necessary Python dependencies via ``pip install -r dev-requirements.txt``, then
run the tests via:

.. code:: shell

  pytest
  ./lint.sh


Submitting a PR
---------------
Please run the tests as described above before submitting a PR.

To add a new app, make a PR that:

- Contains your new example app in its own subdirectory of the ``apps`` subdirectory 
- Adds instructions on running your app to a README.rst in its subdirectory 
- Adds a small description of the app in the root README.rst
- Add tests in `tests` that show that your app works as intended

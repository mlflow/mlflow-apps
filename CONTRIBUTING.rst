How to Contribute
=================

First, install requirements for running the linter via
``pip install -r dev-requirements.txt``

To contribute, make a PR that: \* Contains your new example app in its
own subdirectory of the ``examples`` subdirectory \* Adds instructions
on running your app to a README in its subdirectory \* Adds a small
description of the app root README \* Adds a link to your app’s README
from its description in the root README \* An integration test that
shows that your app works as intended (run pytest from the root
directory) \* Passes linting tests (run ./lint.sh from the root project
directory)
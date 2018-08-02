#!/usr/bin/env bash

set -e

FWDIR="$(cd "`dirname $0`"; pwd)"
cd "$FWDIR"

pycodestyle --max-line-length=100 tests
pylint --msg-template="{path} ({line},{column}): [{msg_id} {symbol}] {msg}" --rcfile="$FWDIR/pylintrc" -- apps tests

rstcheck README.rst

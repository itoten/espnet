#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/eegcls:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh

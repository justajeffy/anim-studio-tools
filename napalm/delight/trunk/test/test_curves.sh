#!/bin/bash
source test_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:`pwd` && testRMan '$@' test_curves' | drd-env -s $NAPALM_DELIGHT_TEST_ENV

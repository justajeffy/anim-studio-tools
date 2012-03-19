#!/bin/bash
source test_env.sh
echo 'drd-context-info && testRMan '$@' test_gx_proc' | drd-env -s $NAPALM_DELIGHT_TEST_ENV

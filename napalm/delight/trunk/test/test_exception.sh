#!/bin/bash
source test_env.sh
echo 'shaderdl my_displace.sl my_surface.sl' | drd-env -s $NAPALM_DELIGHT_TEST_ENV
echo 'drd-context-info && testRMan '$@' test_exception' | drd-env -s $NAPALM_DELIGHT_TEST_ENV

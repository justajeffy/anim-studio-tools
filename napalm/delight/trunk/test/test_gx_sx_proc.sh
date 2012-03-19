#!/bin/bash
source test_env.sh
echo 'shaderdl my_displace.sl my_surface.sl && drd-context-info && testRMan '$@' test_gx_sx_proc' | drd-env -s $NAPALM_DELIGHT_TEST_ENV

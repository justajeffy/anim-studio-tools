#echo 'drd-python test_ptc.py' | drd-env -s napalmDelight
source test_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:`pwd` && shaderdl my_ptc.sl && testRMan '$@' test_ptc' | drd-env -s $NAPALM_DELIGHT_TEST_ENV
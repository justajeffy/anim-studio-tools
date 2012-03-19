#!/bin/bash
trap '{ echo "Hey, you pressed Ctrl-C.  Time to quit." ; exit 1; }' INT

LOGFILE=/tmp/woolLint_test.log

for arg in \
  "test_curves"\
  "test_exception"\
  "test_gx_proc"\
  "test_gx_sx_proc"\
  "test_points" \
  "test_ptc" \
  "test_sphere"

do
  echo 'running test: '$arg

  # remove the old log file
  rm -f $LOGFILE

  # run the test, outputting to the log file
  ./$arg.sh $@ > $LOGFILE 2>&1

  # check the log file for PASS or FAIL
  grep -E "(PASS|FAIL|INFO|what|terminate)" $LOGFILE

  echo '----------------------------'
done


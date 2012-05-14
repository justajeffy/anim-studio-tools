#!/bin/bash

TIMESTAMP=`/bin/cat /var/log/sherman/collector-receipt`

while read e; do export "$e"; done < <(/drd/software/int/bin/launcher.sh --project hf2 --dept rnd --printEnv apps/farmsubmit)

# Submit the job to the farm
#su jay.munzner -c "/drd/software/int/sys/sherman/1.0/bin/farmer.py $TIMESTAMP"
/drd/software/int/sys/sherman/1.0/bin/farmer.py $TIMESTAMP

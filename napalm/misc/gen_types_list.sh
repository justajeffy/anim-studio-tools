#!/bin/bash
#
# Run this script in core/napalm/core/types to print the list of (label, type) pairs that
# appears in Appendix A of the Napalm documentation. When and if new types are added to
# Napalm, use this to keep Appendix A up-to-date.
#
grep _NAPALM_TYPE_OP *.cpp | grep -v def | sed 's/.*(//g' | sed 's/)//g' | sed 's/ //g' | sed 's/\t//g' | tr ',' ' ' | awk '{print $2"\t"$1}'

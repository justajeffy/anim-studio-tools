#!/bin/bash

#echo first argument: $1

FILENAME=$HOME/.config/calib.yaml
TARGET=sRGB

if [ -e $FILENAME ];
then
	LINE=`/usr/bin/head $FILENAME | /bin/grep target`
	if [ "$LINE" == "target: DCI" ];
	then
		TARGET=DCI
	fi
fi
#echo "Target colorspace: $TARGET"
#echo "Exporting LUTs..."
if [ "$TARGET" == "sRGB" ];
then
	export DRD_DISPLAYLUT_RV=$1/share/ocio/hf2/files/Kodak2383CoolGrade_ocio.csp
	export HOUDINI_IMAGE_DISPLAY_LUT=$1/share/ocio/hf2/files/Kodak2383CoolGrade_ocio.lut
else
	export DRD_DISPLAYLUT_RV=$1/share/ocio/hf2/files/Kodak2383CoolGrade_ocio_Dreamcolor.csp
	export HOUDINI_IMAGE_DISPLAY_LUT=$1/share/ocio/hf2/files/Kodak2383CoolGrade_ocio_Dreamcolor.lut
fi
#echo "done."
#echo `/bin/env | /bin/grep DRD_DISPLAYLUT_RV`
#echo `/bin/env | /bin/grep HOUDINI_IMAGE_DISPLAY_LUT`


#!/bin/bash

MY_PATH="`( cd \"$MY_PATH\" && pwd )`"  # absolutized and normalized

if [ -z "$MY_PATH" ] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  echo "PATH FAILURE"
  exit 1  # fail
fi

cd $MY_PATH
cd ../datasets/UCF_Anomaly_Detection

mkdir features
cd features

# SlowFast Features
wget -O Kinetics_c2_SLOWFAST_8x8_R50_NONE_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140087&authkey=AMirV3Fulu9UKj4"
wget -O Kinetics_c2_SLOWFAST_8x8_R50_BG-KNN_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140076&authkey=ABmXmJlleHD0og0"
wget -O Kinetics_c2_SLOWFAST_8x8_R50_BG-MOG2_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140086&authkey=AE139_PjlPADqfA"

# MViT K400 Features
wget -O Kinetics_MVIT_B_32x3_CONV_K400_NONE_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140127&authkey=ANvvAcit-mr8qaE"
wget -O Kinetics_MVIT_B_32x3_CONV_K400_BG-KNN_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140128&authkey=APYAetItQuvgGtY"
wget -O Kinetics_MVIT_B_32x3_CONV_K400_BG-MOG2_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140129&authkey=APDONysZE0xLm9g"

# MViT K600 Features
wget -O Kinetics_MVIT_B_32x3_CONV_K600_NONE_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140124&authkey=ACxVV5fIAZCEzqk"
wget -O Kinetics_MVIT_B_32x3_CONV_K600_BG-KNN_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140125&authkey=AP495bZ_OVEEmic"
wget -O Kinetics_MVIT_B_32x3_CONV_K600_BG-MOG2_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140126&authkey=AHcixzMloYdjqLs"


# MViT Features


unzip \*.zip
rm *.zip

echo "SUCCESS: Features downloaded and fully ready."
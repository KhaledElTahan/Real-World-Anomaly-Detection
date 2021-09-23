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
wget -O Kinetics_c2_SLOWFAST_8x8_R50_BG-KNN_32x32.zip "https://onedrive.live.com/download?cid=43A1E31F48BBE205&resid=43A1E31F48BBE205%2140076&authkey=ABmXmJlleHD0og0"

# MViT Features


unzip \*.zip
rm *.zip

echo "SUCCESS: Features downloaded and fully ready."
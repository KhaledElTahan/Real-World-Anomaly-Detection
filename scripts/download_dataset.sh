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

wget https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABJtkTnNc8LcVTfH1gE_uFoa/Anomaly-Videos-Part-1.zip
wget https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAbdSEUox64ZLgVAntr2WgSa/Anomaly-Videos-Part-2.zip
wget https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAgpsRNSHI_BtRnSCxxR7j9a/Anomaly-Videos-Part-3.zip
wget https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABqY-3fJSmSMafFIlJXRE-9a/Anomaly-Videos-Part-4.zip

wget https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AADEUCsLOCN_jHmmx7uFcUhHa/Training-Normal-Videos-Part-1.zip
wget https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAHZByMMGCVms4hhHZU2pMBa/Training-Normal-Videos-Part-2.zip
wget https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AACeDPUxpB6sY2jKgLGzaEdra/Testing_Normal_Videos.zip

mkdir videos
unzip \*.zip -d videos

mv videos/Training-Normal-Videos-Part-1/ videos/Training_Normal_Videos_Anomaly/
cp -a videos/Training-Normal-Videos-Part-2/. videos/Training_Normal_Videos_Anomaly/
rm -fr videos/Training-Normal-Videos-Part-2/
cp -a videos/Anomaly-Videos-Part-1/. videos/
cp -a videos/Anomaly-Videos-Part-2/. videos/
cp -a videos/Anomaly-Videos-Part-3/. videos/
cp -a videos/Anomaly-Videos-Part-4/. videos/
rm -fr videos/Anomaly-Videos-Part-1/ videos/Anomaly-Videos-Part-2/ videos/Anomaly-Videos-Part-3/ videos/Anomaly-Videos-Part-4/

rm *.zip

echo "SUCCESS: Dataset downloaded and fully ready."

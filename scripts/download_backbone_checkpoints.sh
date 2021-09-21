#!/bin/bash

MY_PATH="`( cd \"$MY_PATH\" && pwd )`"  # absolutized and normalized

if [ -z "$MY_PATH" ] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  echo "PATH FAILURE"
  exit 1  # fail
fi

cd $MY_PATH
cd ../checkpoints

mkdir backbone-checkpoints
cd backbone-checkpoints

mkdir Kinetics
cd Kinetics

wget https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_xs.pyth
wget https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth
wget https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth
wget https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_l.pyth

wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvit/k400.pyth
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvit/k600.pyth

mv k400.pyth k400_MVIT_B_32x3_CONV.pyth
mv k600.pyth k600_MVIT_B_32x3_CONV.pyth

mkdir c2
cd c2

wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/C2D_NOPOOL_8x8_R50.pkl
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_NLN_8x8_R50.pkl

wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_4x16_R50.pkl
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_8x8_R50.pkl

wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl

echo "SUCCESS: Backbone checkpoints downloaded and fully ready."

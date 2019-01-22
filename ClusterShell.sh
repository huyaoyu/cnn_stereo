#!/bin/bash

echo This is the shell script to be running on a cluster.

cd /data/datasets/yaoyuh_tmp/

source /home/yaoyuh/p3/bin/activate

EXE_PYTHON=python

${EXE_PYTHON} /data/datasets/yaoyuh/Projects/cnn_stereo/LocalRunTrainCSN.py --input /data/datasets/yaoyuh/Projects/cnn_stereo/inputTrain_Cluster.json

deactivate

echo Cluster shell ends.


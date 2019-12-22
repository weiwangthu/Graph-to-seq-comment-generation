#!/usr/bin/env bash
python train.py -gpus 0,1,2,3 -use_copy -graph_model GCN -debug

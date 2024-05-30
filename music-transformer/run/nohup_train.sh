#!/bin/bash

project_dir=$( dirname $( pwd ) )

mkdir -p $project_dir/logs
touch $project_dir/logs/{train_output,train_error}.txt

nohup $project_dir/run/run_train.sh > $project_dir/logs/train_output.txt 2> $project_dir/logs/train_error.txt &

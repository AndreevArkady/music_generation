#!/bin/bash

project_dir=$( dirname $( pwd ) )

mkdir -p $project_dir/logs

nohup $project_dir/run/run_preprocess.sh > $project_dir/logs/preprocess_output.txt 2> $project_dir/logs/preprocess_error.txt &

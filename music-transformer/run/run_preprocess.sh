#!/bin/bash

project_dir=$( dirname $( pwd ) )

rm -rf $project_dir/dataset/e_piano/

mkdir $project_dir/dataset/e_piano
mkdir $project_dir/dataset/e_piano/{train,test,val}

source $project_dir/music_transformer_env/bin/activate
cd ..
python3 -m preprocessing.preprocess_midi

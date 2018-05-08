#!/bin/bash

source activate ktf

python write_cnn_h5_file.py \
    --phage-fp /home/jklynch/host/project/viral-learning/data/500_ArcPhage_training_set.fasta \
    --prok-fp /home/jklynch/host/project/viral-learning/data/500_ArcFake_training_set.fasta \
    --output-h5-fp arc_phage_fake_500.h5 \
    --image-width 100 \
    --image-limit 5 &

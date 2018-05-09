#!/bin/sh

#PBS -N 5000_to_h5
#PBS -m bea
#PBS -M jklynch@email.arizona.edu
#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb
#PBS -l walltime=08:00:00

source activate ktf

cd ~/project/viral-learning

python vl/h5/cnn/write_cnn_h5_file.py \
    --phage-fp /rsgrps/bhurwitz/alise/my_data/Riveal_exp/Models/RefSeq_based_models/Prok_Phages_models/size_5000pb/training_set/Phage_trainingset/phage5000_training_set.fa \
    --prok-fp /rsgrps/bhurwitz/alise/my_data/Riveal_exp/Models/RefSeq_based_models/Prok_Phages_models/size_5000pb/training_set/Proc_trainingset/proc5000_training_set.fa \
    --output-h5-fp /extra/jklynch/phage_prok_5000.h5 \
    --image-width 2500 \
    --image-limit 200


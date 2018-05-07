#!/bin/sh

#PBS -N to_h5
#PBS -m bea
#PBS -M jklynch@email.arizona.edu
#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb
#PBS -l walltime=08:00:00

source activate ktf

mkdir -p data/cnn

cd ~/project/viral-learning
python vl/h5/cnn/write_cnn_h5_file.py \
    --phage-fp /rsgrps/bhurwitz/alise/my_data/Riveal_exp/Models/RefSeq_based_models/Prok_Phages_models/size_500pb/training_set/Phage_trainingset/phage500_training_set.fa \
    --prok-fp /rsgrps/bhurwitz/alise/my_data/Riveal_exp/Models/RefSeq_based_models/Prok_Phages_models/size_500pb/training_set/Proc_trainingset/proc500_training_set.fasta \
    --output-h5-fp /extra/jklynch/phage_prok_500.h5 \
    --image-width 100 \
    --image-limit 10


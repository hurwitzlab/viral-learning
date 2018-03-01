#!/bin/sh

#PBS -N to_h5
#PBS -m bea
#PBS -M jklynch@email.arizona.edu
#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb
#PBS -l cput=04:00:00
#PBS -l walltime=02:00:00

source activate ktf

cd ~/project/viral-learning/vl
python to_h5.py

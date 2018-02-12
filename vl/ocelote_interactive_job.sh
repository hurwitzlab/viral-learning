#!/bin/sh

qsub -I -N jobname -m bea -M jklynch@email.arizona.edu -q windfall -l select=1:ncpus=28:mem=168gb -l cput=0:5:0 -l walltime=0:5:0
#!/bin/sh

qsub -I -N interactive-job -m bea -M jklynch@email.arizona.edu -W group_list=bhurwitz -q standard -l select=1:ncpus=28:mem=168gb -l cput=04:00:00 -l walltime=02:00:00

#!/bin/sh

qsub -I -N gpu-vl -m bea -M jklynch@email.arizona.edu -W group_list=bhurwitz -q standard -l select=1:ncpus=28:ngpus=1:mem=168gb -l cput=01:00:00 -l walltime=01:00:00

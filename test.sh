#!/bin/bash
source /home/hpahl/anaconda3/bin/activate
conda activate base

rm -f __pycache__/*
rmdir __pycache__
ls __pycache__ #Make sure its really dead - I trust no one at this point

export NUMBA_DISABLE_INTEL_SVML=0
python _radial_dist.py

#Any text "svml" in cache file?
xxd __pycache__/_radial_dist.boltzmann_radial_potential_linear_density_ebeam-136.py37m.1.nbc | grep svml

export NUMBA_DISABLE_INTEL_SVML=1
python _radial_dist.py

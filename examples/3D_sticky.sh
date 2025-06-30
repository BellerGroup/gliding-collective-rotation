#!/bin/bash

# Active Brownian particles with short-ranged sticky interactions in 3D

/usr/bin/env python3 BDsim.py \
--dim 3 \
--n_chains 2000 \
--chain_length 1 \
--Lx 36 \
--LJ_cutoff_factor 2 \
--T 1e-3 \
--color_by 'z' \
--dtmax 6e-3 \
--v0 3 \
--plot_every 200

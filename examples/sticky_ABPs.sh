#!/bin/bash

# Active Brownian Particle clusters with Lennard-Jones short-range attraction
# and steric repulsion.

/usr/bin/env python3 BDsim.py \
--chain_length 1 \
--bead_density 0.5 \
--T 1e-3 \
--Lx 50 \
--dtmax 2e-2 \
--plot_every 100 \
--v0 3 \
--LJ_cutoff_factor 2 

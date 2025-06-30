#!/bin/bash

# Simulation of gliding assay with mobile motors

/usr/bin/env python3 BDsim.py \
--motor_density 2 \
--D_motors 0.01 \
--motor_max_dist 1 \
#--color_by 'where_active' \
--plot_every 150 \
--chain_length 16 \
--n_chains 20 \
--Lx 32 \
--v0 0.5 \
--dtmax 1e-2 \
--b_one_motor_per_bead False

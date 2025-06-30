#!/bin/bash

# Active Brownian Particle clusters with only steric repulsion interactions.
# Particles form clusters for sufficiently high density and/or low temperature

/usr/bin/env python3 BDsim.py \
--chain_length 1 \
--bead_density 0.5 \
--T 1e-3 \
--Lx 100 \
--dtmax 6e-3 \
--plot_every 200

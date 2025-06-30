#!/bin/bash

# Simulation of Dogic active nematic system: bead-spring chains where beads
# are active only when near an oppositely-oriented bead.
# An external field aligns the chains initially

/usr/bin/env python3 BDsim.py \
--sliding_partner_maxdist 1.25 \
--bead_density 1 \
--v0 0.05 \
--Lx 24 \
--chain_length 20 \
--plot_every 100 \
--kappa 1

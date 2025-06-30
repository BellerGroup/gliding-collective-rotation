#!/bin/bash

# A crystalline packing of passive disks at finite temperature has grain
# boundaries that move over time

/usr/bin/env python3 BDsim.py \
--Lx 50 \
--T 1e-3 \
--bead_density 1.1 \
--chain_length 1 \
--t_init_grow 0.1 \
--v0 0 \
--color_by 'none' \
--dtmax 5e-2 \
--plot_every 100

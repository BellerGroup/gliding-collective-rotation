#!/bin/sh

/usr/bin/env python3 BDsim.py \
--b_display_plots False \
--chain_length 30 \
--plot_every 150 \
--kappa 100 \
--v_tan_angle 0.1 \
--bead_density 0.3 \
--Lx 150 \
--WCA_epsilon 1e-08 \
--WCA_shift 0.1 \
--save_state_every  10 \
--save_every_nth_plot 500 \
--filelabel chiral_test \
--resultspath   ./Results/ \
--imgpath ./Images/ \
> ./Images/screen.out
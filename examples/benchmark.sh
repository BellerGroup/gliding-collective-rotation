#!/bin/bash

# A small run to assess computational speed

/usr/bin/env python3 BDsim.py \
--plot_every -1 \
--ticker_every 20000 \
--Lx 12 \
--chain_length 7 \
--n_threads 2 \
--seed 1 \
--T 0.005

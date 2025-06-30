# gliding-collective-rotation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15776113.svg)](https://doi.org/10.5281/zenodo.15776113)

Brownian dynamics simulation for active Brownian particles and
active bead-spring chains.

This repository is a snapshot of the "Brownian-Dynamics" code base of the Beller Group at Johns Hopkins University, for the purpose of sharing code and scripts used in the manuscript Athani et al., "Gliding microtubules exhibit tunable collective rotation driven by chiral active forces", 2025. 

## Active Brownian dynamics framework

Employs a WCA/Lennard-Jones potential so that beads have steric repulsion
and, optionally, a short-range attraction. Particles on the same chain
additionally have Hookean springs on chain bond lengths and bond angles.
Active self-propulsion has fixed speed and a direction that either
randomly diffuses or is locked to the local tangent of the chain.

Uses a second-order stochastic Runge-Kutta update algorithm. Computation
is CPU-based, with optional multithreading, and accelerated using the just-in-time compilation functionality of Numba (https://numba.pydata.org/).


## Requirements

The following Python packages should be installed before running

- numpy
- numba
- matplotlib
- scipy

For example, you might set up a virtual environment like this: 

```sh
python3 -m venv gliding-collective-rotation-env
source gliding-collective-rotation-env/bin/activate
python3 -m pip install numpy numba matplotlib scipy
chmod +x Chiral_parameters_script.sh
```

Then you will be able to run 
```sh
./Chiral_parameters_script.sh
```

### Works with

- python version >= 3.6.4
- numba version >= 0.50.1

## Usage

Many options are accessible as command line flags. For descriptions of these, type

>		./BDsim.py --help

Example: 

>		./BDsim.py -f 'test' --save_state_every 100 --n_chains 4 --chain_length 7 --Lx 20

The above will run the default setup of flexible bead-spring chains self-propelling along their tangents,
with 4 chains each containing 7 beads (default spacing on chain = 0.5) in a box of size 20x20. By default, the length
scale is set by the beads' repulsion distance. The above line causes the program to save the parameters to 'Results/test_cline.txt',
and to print the locations of all beads to a unique file 'Results/test_[incrementing_number].dat' at intervals of 100 in simulation time
(not number of timesteps).

Other examples can be found in the examples/ folder.

## Initialization

By default, bead-spring chains are initialized at random chain positions and with very small rest-length spacings between consecutive beads on a chain, causing them to appear dot-like in a plot. From there a two-phase procedure is used to generate the initial condition. First, chains retain their small rest-length spacing but their effective hard-core repulsion distance increases linearly from zero. This helps to remove any overlaps. Second, with the repulsion distance saturated, the chains grow, with their linear spring rest-lengths increasing linearly in time. The simulation time "t" value is negative during this procedure, and reaches zero when the "physical" simulation begins.

## Simulation units

Lengths are measured in units of the steric repulsion size r\_WCA = repel_d_max = 2^(1/6) sigma, which is the distance where the (unshifted) WCA/LC potential reaches its minimum. Energy is measured in units of the WCA coefficient epsilon. Time is measured in units of 6 pi eta r\_WCA^3 / epsilon, the reciprocal of the frictional relaxation time of a sphere with radius r\_WCA using the Stokes-Einstein law. For spherical particles of general radius r = r_WCA * r_sim, the damping coefficient gamma becomes the dimensionless size r_sim.

## Analysis tools

The file analysis.py contains some tools for working with saved results from previous runs within the Python intepreter. The analysis.recall() function loads bead positions along with simulation parameters, creating a "beads" Python object (for data) and "sim" Python dictionary (for parameters) that can then be passed to other functions in the package. For example, the bead positions from the third saved timepoint of a run called "test" can be recovered by

>   \>\>\> from analysis import * \
>   \>\>\> recall('test',2)\["beads"\].pos

and the number of chains in that run can be recovered by

>   \>\>\> recall('test',2)\["sim"\]\["n_chains"\]

Images can be constructed directly using two functions that both call the recall() function:

>   \>\>\> snapshot('test',2)

generates a picture of the bead positions at the third saved timepoint of 'test', while

>   \>\>\> animate('test')

generates a sequence of snapshots from all the available timepoints.

## Authors

Written by Daniel Beller & Madhuvanti Athani, 2020-2024.

d.a.beller@jhu.edu, mathani1@alumni.jhu.edu

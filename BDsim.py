#!/usr/bin/env python3

"""
    Template "front-end" file where an arbitrary list of populations can be
    defined and given custom interactions. Parameters can be changed here
    to override command-line inputs, either globally or for individual populations.
    This information is then passed to control.py which runs the simulation.
"""

import numpy as np
import populations, cline, control, updates, config
sim = cline.cline()

################################################################################

                ### Optionally override global parameters here ###

config.configure(sim)

                    ### Define populations here ###

pops = [ populations.Equal_Size_Chain_Collection(sim, name = "beads") ]

if sim['n_motors'] > 0:
    pops += [ populations.Motor_Population(sim) ]

################################################################################

control.run_sim(sim,pops)

#!/usr/bin/env python3

""" Example modification of the template script file including multiple populations with a customized interaction. """

import numpy as np
import populations, cline, control, updates, config
sim = cline.cline()

################################################################################

                ### Optionally override global parameters here ###

sim['dim'] = 3
sim['Lx'] = 50
sim['color_by'] = 'chain_id'
# prevent automatic n_grid = 1 (slow!) due to big sphere:
sim['n_grid_min'] = 5
config.configure(sim)

                    ### Define populations here ###

box_center = np.full(config.dim, config.Lx/2)
# Steve is a single sphere
Steve = populations.Equal_Size_Chain_Collection(
    sim, chain_length = 1, num_chains = 1, name = "Steve")
Steve.v0 = 0 # Steve doesn't self-propel
Steve.set_size(20) # Steve is big
Steve.pos[0] = box_center # Steve is initialized at the center of the box
Steve.initialized = True # Don't randomize Steve's initial position
Steve.fixed = True # Don't update Steve's position

# Kimberly is a collection of longer, self-propelling chains that create a
# sticky interaction with chains from Steve
Kimberly = populations.Equal_Size_Chain_Collection(
    sim, chain_length = 10, num_chains = 50, name = "Kimberly")

## Replace the usual WCA interaction with a sticky interaction
# width of potential
interaction_repel_d_max = 3 * Kimberly.repel_d_max
# a compromise definition of hard-core repulsion distance
mutual_repulsion_radius = 0.5 * (Steve.repel_d_max + Kimberly.repel_d_max)

Kimberly.add_special_interaction("Steve", "WCA",
    WCA_shift = mutual_repulsion_radius - interaction_repel_d_max, # shift zero so that potential minimum is at r = mutual_repulsion_radius
    repel_d_max = interaction_repel_d_max, # equilibrium separation relative to WCA_shift
    LJ_cutoff_factor = 2.5 * interaction_repel_d_max, # activates attractive region of Lennard-Jones potential farther than repel_d_max away from r = WCA_shift
    WCA_epsilon = 3 # strength of potential is stronger than default WCA
)

#
# # make the force reciprocal between Steve and Kimberly
Steve.special_interactions["Kimberly"] = Kimberly.special_interactions["Steve"]

Kimberly.random_initialization()


pops = [Steve, Kimberly]


################################################################################

control.run_sim(sim,pops)

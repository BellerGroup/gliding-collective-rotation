#!/usr/bin/env python3

# Example modification of the template front-end file including multiple populations
# with a customized interaction.

import numpy as np
import populations, cline, control, updates, config
sim = cline.cline()

################################################################################

                ### Optionally override global parameters here ###

sim['Lx'] = 50
sim['color_by'] = 'population'
config.configure(sim)

                    ### Define populations here ###

# Steve is a collection of short chains with no self-propulsion
Steve = populations.Equal_Size_Chain_Collection(
    sim, chain_length = 2, num_chains = 100, name = "Steve")
Steve.v0 = 0 # no self-propulsion

# Kimberly is a collection of longer, self-propelling chains that create a
# sticky interaction with chains from Steve
Kimberly = populations.Equal_Size_Chain_Collection(
    sim, chain_length = 10, num_chains = 30, name = "Kimberly")

# Kimberly and Steve stick to each other
Kimberly.add_special_interaction("Steve", "WCA",  LJ_cutoff_factor=2 )
Steve.special_interactions["Kimberly"] = Kimberly.special_interactions["Steve"]

pops = [Kimberly, Steve]


################################################################################

control.run_sim(sim,pops)

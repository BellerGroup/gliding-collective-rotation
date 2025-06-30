#!/usr/bin/env python3

# Note: default value type (float, int, string, etc.) signals variable type of
# the property, so always use a decimal point (e.g. 4.0) for float-valued properties.
import numpy as np
from time import time as clocktime

"""
Define the command-line options, giving them default values and help strings.
"""

def set_defaults():
    sim = {} # dictionary to hold all option values
    help = {} # dictionary to hold help messages

    sim['filelabel'] = 'most_recent'
    help['filelabel'] = 'Unique label string for each run.'

    ### Numba options
    sim['b_cached'] = True
    help['b_cached'] = 'Use cached compiled functions (True) or recompile (False).'

    sim['n_threads'] = 3
    help['n_threads'] = 'Number of threads used to calculate forces.'

    ### system setup ###
    sim['dim'] = 2
    help['dim'] = 'System dimensionality (2 or 3).'

    sim['SRK_order'] = 2
    help['SRK_order'] = 'Order of Runge-Kutta update (2 or 4).'

    sim['Lx'] = 50.
    help['Lx'] = 'Size of system in both x and y directions.'

    sim['walls'] = 'none'
    help['walls'] = 'Boundary conditions: replace periodic with WCA. Example: pass string "xz" for walls in x and z directions.'

    sim['bead_density'] = 0.5
    help['bead_density'] = 'Initialization: Number of beads per unit square.'

    sim['chain_length'] = 10
    help['chain_length'] = 'Number of beads on each chain'

    sim['n_chains'] = -1
    help['n_chains'] = 'Initialization: Number of chains. If positive, overrides bead_density; otherwise not used.'

    ### stopping conditions ###
    sim['t_max'] = -1
    help['t_max'] = 'Stopping conditions: maximum time. Set negative for infinity (unused).'

    sim['stepnum_max'] = -1
    help['stepnum_max'] = 'Stopping conditions: maximum step number. Set negative for infinity (unused).'

    ### update rules ###
    sim['dtmax'] = 4.e-3 # maximum timestep size
    help['dtmax'] = 'Maximum timestep size (before multiplication by RK_cheat_factor).'

    sim['dt0'] = 0.0001 * sim['dtmax']
    help['dt0'] = 'Small timestep to use in first update only, before we have our first measurement of the forces.'

    sim['RK_cheat_factor'] = 1.
    help['RK_cheat_factor'] = 'For Runge-Kutta update, take a step this many times larger than the test step size used for calculating forces. (Probably not kosher but seems to work when set at 4 if used alongside adaptive timestep.) The benefit is negligible in dense systems but can be significant in dilute systems where many timesteps have no large forces.'

    sim['timestep_safety_factor'] = 0.025
    help['timestep_safety_factor'] = 'For adaptive timestep: Fraction of smallest physical lengthscale allowed as an update distance for any particle. Used to set dt when any particle experiences a large force. Smaller is safer, larger is more efficient. Typically must be decreased for larger spring stiffnesses or velocities.'

    sim['n_grid_max'] = 0
    help['n_grid_max'] = 'Grid/neighbor lists: Maximum number of grid cells per side. Set to 0 to not impose any limit; set to 1 to not use neighbor lists.'

    sim['n_grid_min'] = 1
    help['n_grid_min'] = 'Grid/neighbor lists: Minimum number of grid cells per side. Overrides automatic calculation based on biggest interaction distance.'

    sim['grid_factor'] = 2.
    help['n_grid_factor'] = 'Grid/neighbor lists: Safety factor for deciding grid spacing.'

    sim['T'] = 1e-3
    help['T'] = 'Boltzmann\'s constant times temperature, k_B T'

    ### Forces ###

    ## steric repulsion ##

    sim['repel_d_max'] = 1.
    help['repel_d_max'] = 'WCA: Steric repulsion diameter of beads (before shift by WCA_shift). By default, sets length scale. Equal to 2^(1/6) times sigma, as the WCA/LJ potential is usually written.'

    sim['WCA_epsilon'] = 1.
    help['WCA_epsilon'] = 'WCA: Strength of potential. By default, sets force scale.'

    sim['WCA_shift'] = 0.0 * sim['repel_d_max']
    help['WCA_shift'] = 'WCA: Shift of dependence on distance, V(r) = V_WCA(r + WCA_shift) if r + WCA_shift >= 0, otherwise 0. Larger (positive) values soften potential, allowing crossovers.'

    sim['LJ_cutoff_factor'] = 1.
    help['LJ_cutoff_factor'] = 'WCA: Cutoff distance for potential, before shift by WCA_shift. Set to 1 for purely repulsive WCA. Set >= 2 to include LJ attraction.'

    sim['n_nearby_beads_dont_repel'] = 4
    help['n_nearby_beads_dont_repel'] = 'WCA: Turn off WCA between beads separated by <= this many bead postions on the same chain. Set < 0 to never apply WCA to beads on the same chain.'

    ## linear springs ##
    sim['b_FENE'] = False
    help['b_FENE'] = 'Linear springs:  Replace harmonic potential with nonlinear FENE potential.'

    sim['k'] = 100.
    help['k'] = 'Linear springs:  Spring constant.'

    sim['a'] = 0.5
    help['a'] = 'Linear springs: Rest length.'

    ## bending springs ##
    sim['kappa'] = 1e1
    help['kappa'] = 'Bending springs: bending rigidity.'

    sim['spontaneous_bond_angle'] = 0.0 * np.pi
    help['spontaneous_bond_angle'] = 'Bending springs: Offset angle to zero of potential, causing spontaneous curvature to the right if < 0 or left if > 0.'

    sim['rigid_rod_projection_factor'] = 0.
    help['rigid_rod_projection_factor'] = 'Updates are linearly interpolated with the update that would be experienced by a rigid rod. Set to 1. for rigid rods.'

    sim['FENE_tolerance'] = 0.1
    help['FENE_tolerance'] = 'Linear springs:  Maximum extension from rest length allowed by FENE potential, as fraction of rest length.'

    ## active self-propulsion  ##
    sim['v0'] = 1.
    help['v0'] = 'Self-propulsion: Force (proportional to speed for isolated particles).'

    sim['v_tan_angle'] = 0.0 * np.pi
    help['v_tan_angle'] = 'Self-propulsion: Angle offset of active force relative to tangent.'

    sim['b_use_tangent_as_propulsion_direction'] = True
    help['b_use_tangent_as_propulsion_direction'] = 'Self-propulsion: Chains typically propel along tangent (True). If (False) or if chain_length = 1, then self-propulsion direction instead undergoes rotational diffusion.'

    sim['sliding_partner_maxdist'] = 0.
    help['sliding_partner_maxdist'] = 'Self_propulsion: For sliding-filament active nematics, allow self-propulsion only for beads within this distance of some other bead with orientation > 90 degrees different. If <= 0, not used i.e. beads always self-propel.'

    sim['t_wait_v0'] = -1.
    help['t_wait_v0'] = 'Early times: Wait until this time before turning on self-propulsion. Not used if <= 0.'

    ## Gaussian noise ##
    sim['seed'] = int(clocktime())
    help['seed'] = 'Noise: Seed for random number generator for reproducible runs.'

    sim['D'] = - 1.
    help['D'] = 'Noise: Translational diffusion constant for particles with nonzero size r. If negegative, defaults to kT/r.'

    sim['Dr'] = -1.
    help['Dr'] = 'Noise: Rotational diffusion constant. If negative, defaults to 3 D / (4 repel_d_max^2). Used only if chain_length = 1 or b_use_tangent_as_propulsion_direction == True.'

    ### Initialization values  ###

    sim['pos_init_noise'] = 1.
    help['pos_init_noise'] = 'Initialization: Translational noise about center of system, scaled by system size. (Uniform distribution requires whole number.)'

    sim['phi_init'] = 0.
    help['phi_init'] = 'Initialization: Reference initial angle for all chains (before randomization).'

    sim['phi_init_noise'] = 1.e5
    help['phi_init_noise'] = 'Initialization: Angular noise for randomization of initial chain angles.'

    sim['a_init_factor'] = 1.e-8
    help['a_init_factor'] = 'Initialization: Shrink chain rest lengths by this fraction, before growing them to regular length.'

    sim['t_init_grow'] = 1.
    help['t_init_grow'] = 'Initialization (growing phase): Time interval for chains to grow to their regular length. Time is initially set to negative of this.'

    sim['Efield_strength_grow'] = 0.
    help['Efield_strength_grow'] = 'Initialization (growing phase): While growing, apply a field of this strength that aligns rods with the box diagonal.'

    sim['b_grow'] = False
    help['b_grow'] = 'Growth: Whether chains should grow and divide (True) or not (False).'

    sim['grow_rate'] = 0.2
    help['grow_rate'] = 'Growth: If b_grow == True, rate at which rest length grows as fraction of \'reference\' rest length.'

    sim['grow_rate_noise'] = 0.1
    help['grow_rate_noise'] = 'Growth: Random component of growth rate; growth length each timestep is multiplied by (1 + grow_rate_noise * \\xi) where \\xi is a uniform random number in [-1,1].'

    ### motors

    sim['n_motors'] = 0
    help['n_motors'] = 'Motors: total number. If <= 0, bead self-propulsion is independent of position; otherwise requires a motor nearby.'

    sim['motor_density'] = 0.
    help['motor_density'] = 'Motors: area density. If > 0, overrides n_motors.'

    sim['D_motors'] = 0.1
    help['D_motors'] = 'Motors: translational diffusion constant'

    sim['motor_max_dist'] = sim['a'] / 2
    help['motor_max_dist'] = 'Motors: Maximum distance for a motor to \'push\' a bead.'

    sim['b_motors_dont_diffuse_while_pushing'] = True
    help['b_motors_dont_diffuse_while_pushing'] = 'Motors: Prevent motors from diffusing during timesteps when they push a bead (True), or let them diffuse as normal (False).'

    sim['b_one_motor_per_bead'] = False
    help['b_one_motor_per_bead'] = 'Motors: Bead self-propulsion is binary [0, v0] (True) or additive [ n * v0 ] (False) with respect to number n of nearby motors.'

    sim['k_on'] = 0.9
    help['k_on'] = 'Motors: k_on rate of a motor.'

    sim['k_off'] = 0.2
    help['k_off'] = 'Motors: k_off rate of a motor'

    ### output options ###

    ## Ticker ##

    sim['ticker_every'] = 1000 # print some info to stout every this many steps
    help['ticker_every'] = 'Interval (in simulation steps) for printing status info to standard out.'

    ## Save state to file ##

    sim['resultspath'] = './Results/'
    help['resultspath'] = 'Path to results folder.'

    sim['save_state_every'] = -1.
    help['save_state_every'] = 'Time (not step number) interval between outputs to save state. Set negative to never save.'

    sim['b_save_most_recent_only'] = False
    help['b_save_most_recent_only'] = 'Repeatedly overwrite a single file (True) or write each saved state to a separate, sequentially labeled file (False).'

    sim['n_digits_stepnum_str'] = 8
    help['n_digits_stepnum_str'] = 'Number of digits in numerical labels at the ends of image and state filenames; small numbers have zeros appended to the left.'

    ## Plots ##

    sim['plot_every'] = 500
    help['plot_every'] = 'Plots: Update interval in simulations steps. If <= 0, plots are never made.'

    sim['figsize'] = 10
    help['figsize'] = 'Plots: Figure size.'

    color_by_choices = ['chain_orientation', 'chain_director', 'bead_orientation',
                        'chain_id', 'order_on_chain', 'where_active', 'inherited_label']
    sim['color_by'] = color_by_choices[0]
    help['color_by'] = 'Plots: How to color beads. Choices: ' + ', '.join(color_by_choices)

    sim['imgpath'] = './Images/'
    help['imgpath'] = 'Path to folder where images are saved.'

    sim['b_save_most_recent_img_only'] = False
    help['b_save_most_recent_img_only'] = 'Repeatedly overwrite a single image file (True) or write each image to a separate, sequentially labeled file (False).'

    sim['save_every_nth_plot'] = -1
    help['save_every_nth_plot'] = 'Number of simulation steps between image outputs. Set negative to never save images.'

    sim['b_display_plots'] = True
    help['b_display_plots'] = 'Interactive plots (True) or only save plots to files (False).'

    sim['b_show_colorbar'] = True
    help['b_show_colorbar'] = 'Plots: show color bar legend.'

    sim['growing_stepskip'] = 10
    help['growing_stepskip'] = 'Steps to undergo at same length while growing.'

    return sim, help

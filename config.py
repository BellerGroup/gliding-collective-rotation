#!/usr/bin/env python3

import numba as nb
import numpy as np
import os
import config, populations
from time import sleep

"""
Utility module for common variables needed by several modules
"""

b_cached = True
sliding_partner_maxdist = 0.
n_threads = 1
dim = 2
Lx = 1.
config.n_grid = 1
b_FENE = False
pop_name_list = []
b_announcements = True
n_steps_update_grid_list = 1
RK_cheat_factor = 1
max_dist_per_step = 1
grid_factor = 2.
walls = False

def std_numba_kwargs():
    return {
        'nopython':True,
        'parallel':False,
        'fastmath':True,
        'cache':config.b_cached,
        'nogil':True
    }

class status_class:
    """ General time-keeping object """
    def __init__(self, sim, start_t = None, start_stepnum = 0, start_filenumber = 0):
        if start_t == None:
            self.t_start = - sim['t_init_grow']
        else:
            self.t_start = start_t

        self.stepnum = start_stepnum
        self.filenumber = start_filenumber

        self.SRK2_step = 1
        self.b_calc_active_forces = True
        self.reset_timekeeping(sim)

    def reset_timekeeping(self, sim):
        self.t = self.t_start
        self.dt = sim['dt0']
        self.update_nborlists_count = 0
        self.update_mtr_nborlists_count = 0
        # self.max_force_strength_sum = 0
        self.max_forces_strength = 0
        self.t_prev = 0 # dummy initialization
        self.clocktime_prev = 0 # dummy initialization
        self.dt_used = 0 # dummy initialization
        self.next_save_time = self.t_start
        self.next_output_step = 0
        self.update_grid_list_counter = 0
        self.dtmax = sim['dtmax']
        self.prev_ticker_step = 0

    def dt_SRK2(self):
        if self.SRK2_step == 1:
            return self.dt
        else:
            return self.dt_used


def configure(sim):
    """ Some further global variables derived from command-line parameters """

    sim['epsilon'] = 1e-8 # reference small number; smaller than this is considered zero

    # a chain takes up excluded volume repel_d_max * ( a * (chain_length-1) + repel_d_max
    sim['chain_density'] = sim['bead_density'] / (sim['repel_d_max'] * \
                    ( sim['a'] * (sim['chain_length']-1) + sim['repel_d_max']) )
    if sim['n_chains'] < 0:
        sim['n_chains'] = max(1,round(sim['Lx']*sim['Lx']*sim['chain_density'])) # number of chains
        # otherwise use positive value of n_chains entered in command line
    sim['n_beads'] = sim['chain_length'] * sim['n_chains'] # total number of beads

    if sim['motor_density'] > 0:
        sim['n_motors'] = int(
            round(sim['Lx'] * sim['Lx'] * sim['motor_density'])
        )

    sim['plot_pause_time'] = sim['epsilon'] # nominal pause for plot update

    if sim['n_motors'] > 0:
        sim['motor_imparts_force'] = sim['v0']
        sim['v0'] = 0
        print(f'----Self-propulsion speed v0 = {sim["motor_imparts_force"]:.3e} transferred to motors\' action on beads.')
    else:
        sim['motor_imparts_force'] = 0

    if sim['Dr'] < 0 and sim['D'] > 0:
        sim['Dr'] = 3 * sim['D'] / (4 * sim['repel_d_max']*2) # Stokes-Einstein relation

    sim['outfileprefix'] = sim['resultspath'] + sim['filelabel']
    sim['imgfileprefix'] = sim['imgpath'] + sim['filelabel']


    config.dim = sim['dim']
    config.SRK_order = sim['SRK_order']
    config.Lx = sim['Lx']
    config.walls = np.array(
        [
            (st in sim['walls'] or sim['walls'] in ['all','True'])
            for st in ['x','y','z']
        ]
    )

    np.random.seed(sim['seed'])

    nb.set_num_threads(1)

    config.b_cached = sim['b_cached']

    dim_filename = '__pycache__/BDsim_dim.dat'
    if os.path.exists(dim_filename):
        with open(dim_filename,'r') as f:
            prev_dim = int(f.readlines()[0].split()[0])
        if prev_dim != sim['dim']:
            config.b_cached = False
            print('--Previous run had different dimension. Numba-accelerated functions must (re)compile.')
    else:
        config.b_cached = False
        print('--Previous run not detected. Numba-accelerated functions must (re)compile.')
    if not sim['b_cached']:
        print('--Recompilation requested by user.')
    if not config.b_cached:
        print('----Deleting contents of __pycache__/ and (re)compiling.')
        sleep(2)
        os.system('rm __pycache__/*')
        with open(dim_filename,'w') as f:
            f.write(str(sim['dim']))

    return


def grid_list_setup(sim, pops):

    # get length scales that may affect grid spacing and update timing
    max_interact_dists_by_pop = [
        max([
            2*max(pop.rest_lengths)
            if isinstance(pop, populations.Chain_Collection) and pop.longest_chain_length > 1
            else 0,
            max(
                pop.WCA_params[-1], # <- LJ_cutoff_distance
                pop.sliding_partner_maxdist
            )
            if isinstance(pop, populations.WCA_Population)
            else 0,
            pop.motor_max_dist
            if isinstance(pop, populations.Motor_Population)
            else 0
        ]) for pop in pops
    ]

    sim['biggest_interaction_distance'] = max(max_interact_dists_by_pop)
    # beads separated by distances larger than this are assumed to have no
    # interactions

    sim['n_grid'] = max(
        sim['n_grid_min'],
        int(sim['Lx']/(sim['grid_factor']*sim['biggest_interaction_distance']))
    )
    if sim['n_grid_max'] > 0:
        sim['n_grid'] = min(sim['n_grid'], sim['n_grid_max'])

    sim['grid_spacing'] = sim['Lx']/sim['n_grid']
    grid_relevant_max_interact_dists = [
        dist for dist in max_interact_dists_by_pop if dist < sim['Lx'] / 2
    ]
    sim['n_steps_update_grid_list'] = max(
        1,
        int(1. / (sim['timestep_safety_factor']*sim['RK_cheat_factor']))
    )
    # if two particles are barely out of range to be grid neighbors,
    # what is max dist they can each move between grid updates with guarantee
    # that they don't come within their interaction distance?
    sim['max_dist_per_step'] = 0.5*(sim['grid_factor'] - 1) * min(grid_relevant_max_interact_dists) / sim['n_steps_update_grid_list']

    sim['grid_window'] = int(
        np.ceil(sim['biggest_interaction_distance']/sim['grid_spacing'])
    )

    config.n_grid = sim['n_grid']
    config.grid_factor = sim['grid_factor']
    config.n_steps_update_grid_list = sim['n_steps_update_grid_list']
    config.max_dist_per_step = sim['max_dist_per_step']
    for pop in pops:
        pop.make_empty_grid_list(num_pops = len(pops))

    ## Warnings

    effective_chain_length = (
        max([
            (pop.longest_chain_length - 1) * pop.reference_rest_length
            if isinstance(pop, populations.Chain_Collection)
            else 0
            for pop in pops
        ]) + sim['biggest_interaction_distance']
    )
    if 2 * effective_chain_length > sim['Lx'] and not sim['b_FENE']:
    	print(f'\n~~~WARNING: effective chain length ≈ {effective_chain_length} is not much smaller than system size {sim["Lx"]}, which may cause errors.')
    	print(f'~~~~~~Recommendation: Change Lx to at least {2*effective_chain_length:.2f}.')

    if sim['n_nearby_beads_dont_repel'] >= 0 and sim['a'] * sim['n_nearby_beads_dont_repel'] < sim['repel_d_max'] and sim['chain_length'] != 1:
    	print(f'\n~~~WARNING: beads at rest distance on same chain (not necessarily consecutive) will repel.')
    	print(f'~~~~~~Recommendation: Increase spring rest length (a) or n_nearby_beads_dont_repel such that their product is > repel_d_max = {sim["repel_d_max"]:.3f} (current value: {sim["a"] * sim["n_nearby_beads_dont_repel"]:.3f}).')


    if np.any(config.walls):
        max_safe_pos_init_noise = 1.-2.5*sim["grid_window"]/sim["n_grid"]
        if sim['pos_init_noise'] > max_safe_pos_init_noise:
            print(f'~~~~WARNING: pos_init_noise={sim["pos_init_noise"]} likely to initialize particles inside walls or unphysically close to them.')
            print(f'~~~~~~Recommendation: reduce pos_init_noise to at most {max_safe_pos_init_noise}.') #?? why 2.5?


#!/usr/bin/env python3

import output_methods as io
import updates, populations, config
import math
import numpy as np

"""
The simulation loop and associated high-level functions for configuration and
exiting.
"""

def run_sim(sim, pops, start_t = None, start_stepnum = 0, start_filenumber = 0):
    """ Highest-level function after front-end file. """
    status = config.status_class(sim, start_t = start_t,
        start_stepnum = start_stepnum, start_filenumber = start_filenumber)
    configure_start(sim, pops, status)
    start_msgs(sim) # Some helpful messages before we start
    starting_sim_msgs()
    sim_loop(pops, sim, status)
    exit_msgs(status)

def sim_loop(pops, sim, status):

    if sim['plot_every'] > 0:
        figure, beads_plots, motors_plots = io.create_plot(sim,pops)
        prev_plot_step = 0

    while not ( (status.stepnum > sim['stepnum_max'] >= 0) or (status.t > sim['t_max'] >= 0 ) ):

        if sim['b_init_growing'] or sim['b_waiting_v0']:
            early_time_procedures(pops, status, sim)
        status.next_output_step = -1
        if sim['plot_every'] > 0:
            if (
                status.stepnum==0
                or status.stepnum-prev_plot_step >= sim['plot_every']
            ):
                io.update_plot(sim,pops,status,figure,beads_plots,motors_plots)
                prev_plot_step = status.stepnum
            status.next_output_step = prev_plot_step + sim['plot_every']

        ## print status to terminal
        if sim['ticker_every'] > 0:
            if status.stepnum == 0:
                io.first_ticker(sim,status)
                status.prev_ticker_step = 0

            elif status.stepnum - status.prev_ticker_step >= sim['ticker_every']:
                io.ticker(sim, status)
                status.prev_ticker_step = status.stepnum

            status.next_output_step = min([
                item for item in [
                    status.next_output_step,
                    status.prev_ticker_step + sim['ticker_every']
                ] if item >= 0
            ])

        ## print positions and orientations to file
        if sim['save_state_every'] > 0 and (status.stepnum == 0 or status.t >= status.next_save_time):
            io.save_state(sim,pops,status)
            if status.t >= status.next_save_time:
                status.next_save_time += sim['save_state_every']

        # if growing, need to exit jitted loop frequently
        if sim['b_init_growing']:
            status.next_output_step = status.stepnum + min([
                item for item in [
                    sim['growing_stepskip'], sim['ticker_every'], sim['plot_every']
                ] if item > 0
            ])

        ## the update step
        for pop in pops:
            # check if this is a growing (not the same as init_grow!) population
            if isinstance(pop, populations.Chain_Collection):
                if pop.b_grow:
                    pop.grow_and_perhaps_divide_all_chains(sim, status)
                    status.next_output_step = status.stepnum + 1
                    status.n_steps_update_grid_list = 1

        updates.SRK2_force_calc(pops, status, sim)


def start_msgs(sim):
    s = f'--Creating {sim["Lx"]} x {sim["Lx"]}'
    if config.dim == 3:
        s += f' x {sim["Lx"]}'
    s += f' simulation domain with periodic boundary conditions'
    print(s)
    s = f'--Using {sim["n_grid"]} x {sim["n_grid"]}'
    if config.dim == 3:
        s += f' x {sim["n_grid"]}'
    s += f' cell list updated every {sim["n_steps_update_grid_list"]} timesteps'
    print(s)
    if sim['save_state_every'] > 0:
        print("--Results will be saved to " + sim['outfileprefix'] + " ... .dat")
    else:
        print("--Not saving results")
    return


def starting_sim_msgs():
    print("--Starting simulation... here we go!")
    print('----Most functions compile just-in-time now if not cached. This may take a moment.')
    print('----Initializing positions.')
    return


def exit_msgs(status):
    print(f'Quit at step {status.stepnum}, t={status.t:.3f}')


def configure_start(sim, pops, status):
    config.pop_name_list = []
    for pop in pops:
        if pop.name in config.pop_name_list:
            pop.name = pop.name + '_2'
        config.pop_name_list.append(pop.name)

    for pop in pops:
        if not pop.initialized:
            pop.random_initialization()

    config.grid_list_setup(sim,pops)
    for pop in pops:
        pop.update_grid_list(overwrite = True)
    for pop1 in pops:
        if isinstance(pop1, populations.Forces_Population):
            for pop2 in pops:
                pop1.update_nbor_list_with_pop2(
                    pop2, grid_window = sim['grid_window']
                )
    if sim['t_init_grow'] > 0 and status.t_start < 0:
        for pop in pops:
            if isinstance(pop, populations.Chain_Collection):
                pop.set_init_grow()
        sim['b_init_growing'] = True
    else:
        sim['b_init_growing'] = False

    if sim['t_wait_v0'] > 0:
        print(f'----Turning off active self-propulsion until t={sim["t_wait_v0"]}')
        sim['b_waiting_v0'] = True
    else:
        sim['b_waiting_v0'] = False

    config.n_steps_update_grid_list = sim['n_steps_update_grid_list']


def early_time_procedures(pops, status, sim):
    """ These items are checked within the simulation loop and are only relevant
        during the early stages of the simulation.
    """

    if sim['b_waiting_v0']:
        if status.t > sim['t_wait_v0']:
            sim['b_waiting_v0'] = False
            if sim['t_wait_v0'] > 0:
                print(f'----Turning on active self-propulsion')
            if not sim['b_init_growing']:
                status.b_calc_active_forces = True

    if sim['b_init_growing']:
        if status.t > 0:
            sim['b_init_growing'] = False
            if sim['t_init_grow'] > 0:
                print(f'----Initial growing phase complete')
            status.b_calc_active_forces = not sim['b_waiting_v0']
            config.RK_cheat_factor = sim['RK_cheat_factor']

            for pop in pops:
                if isinstance(pop, populations.WCA_Population):
                    pop.Efield *= 0.
                if isinstance(pop, populations.Chain_Collection):
                    pop.end_init_grow()
        else:
            status.b_calc_active_forces = False
            config.RK_cheat_factor = 1
            for pop in pops:
                if isinstance(pop, populations.Chain_Collection):
                    delta_t = status.t - status.t_start
                    init_grow_progress = max(delta_t/sim['t_init_grow'], sim['epsilon'])
                    if pop.longest_chain_length == 1:
                        init_repel_frac = 1
                        grow_portion_frac = 0
                        repel_portion_frac = 1
                    else:
                        init_repel_frac = 0.25
                        init_rigid_frac = 1.
                        grow_portion_frac = max(0,
                            (init_grow_progress - init_repel_frac)
                            /(1 - init_repel_frac)
                        )
                        repel_portion_frac = min(
                            init_repel_frac, init_grow_progress
                        ) / init_repel_frac
                        pop.repel_d_max = pop.repel_d_max_save
                        pop.rigid_rod_projection_factor = max(0.025, pop.rigid_rod_projection_factor_save)
                        if init_grow_progress < init_repel_frac:
                            pop.rigid_rod_projection_factor = 0
                            pop.repel_d_max = pop.repel_d_max_save * repel_portion_frac
                        elif init_grow_progress < init_rigid_frac:
                            # pop.rigid_rod_projection_factor = 1.
                            pop.repel_d_max = pop.repel_d_max_save
                            pop.kappa = 10
                        else:
                            pop.rigid_rod_projection_factor = 0.
                            pop.kappa = pop.kappa_save
                            pop.k = pop.k_save
                    pop.adjust_WCA_params()
                    init_grow_exponent = 0.1 # empirically chosen
                    pop.rest_lengths[:] = pop.reference_rest_length * (
                         sim['a_init_factor']**(1-init_grow_progress**init_grow_exponent)
                    )
                    if grow_portion_frac > 0 and sim['Efield_strength_grow'] > 0:
                        pop.Efield = sim['Efield_strength_grow'] * np.array([1 if i==1 else 0 for i in range(config.dim)]) / np.sqrt(config.dim)

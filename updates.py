#!/usr/bin/env python3

import numpy as np
from numba import jit, float64, int64, boolean
import populations, config, forces, SRK
import threading
import time



## LLVM debug mode:
# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')

"""
Interfaces between the Population-derived classes and the force calculations
in forces.py, as well as updating the cell lists and neighborlists.
"""


def make_multithread(inner_func, pops_args):
    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """

    # Adapted from multithreading example at
    # https://numba.pydata.org/numba-doc/dev/user/examples.html

    # Note: We are doing "manual multithreading" here, which requires the
    # nogil=True option in the jitted function to be truly parallel. We
    # are not using the automatic parallelization provided by Numba as we
    # haven't found cases where it improves performance.

    # If using only one thread, go directly to calculation
    if config.n_threads == 1 and len(pops_args) == 1 and len(pops_args[0]) == 1:
        def func_single_threaded(pops_args):
            _, chunkable_args, unchunkable_args = pops_args[0][0]
            inner_func(*((np.array([0,0,0,1,1]),) + chunkable_args + unchunkable_args))
        return func_single_threaded(pops_args)

    else: # set up multithreading
        def func_multi_threaded(pops_args):
            threads = []
            thread_num = 0

            # arguments received in a ( pop x other_pop ) list-array
            for pop_i, pop_args_list in enumerate(pops_args):

                # Update positions in thread where other_pos is pos
                pop_name_index = pop_args_list[0][2][0][0]
                other_pop_name_indices = [
                    item[2][0][1] for item in pop_args_list
                ]
                self_pop_j = other_pop_name_indices.index(pop_name_index)
                # find split_indices for population's self-interactions
                split_indices_self = pop_args_list[self_pop_j][0]
                # calculate # of threads for this population's self-interactions
                if isinstance(split_indices_self, int): # we passed number of threads instead of split indicces
                    num_threads_this_pop_self = split_indices_self
                elif isinstance(split_indices_self, np.ndarray):
                    num_threads_this_pop_self = len(split_indices_self) + 1
                else:
                    raise ValueError

                pop_j = 0 # index of other_pop
                for split_indices, chunkable_args, unchunkable_args in pop_args_list:

                    if isinstance(split_indices, int):
                        # if number of threads provided instead of split_indices, automatically chunk the chunkable arguments
                        num_threads_this_pop_pair = split_indices
                        array_split_arg_2 = num_threads_this_pop_pair
                    elif isinstance(split_indices, np.ndarray):
                        num_threads_this_pop_pair = len(split_indices) + 1
                        array_split_arg_2 = split_indices
                    else:
                        raise ValueError

                    # chunk the chunkable arguments
                    chunks_T = [ np.array_split(arg, array_split_arg_2) if len(arg) > 0 else [arg]*num_threads_this_pop_pair for arg in chunkable_args ]

                    chunks = [ [ item[i] for item in chunks_T ] for i in range(num_threads_this_pop_pair) ]

                    start_indices = np.append([0],np.cumsum( [len(split_args[0]) for split_args in chunks] )[:-1])

                    other_pop_name_index = unchunkable_args[0][1]

                    # Set up one thread per chunk, per other_pop, per population
                    for thread_num_this_pop_pair, start_index, chunk in zip(range(num_threads_this_pop_pair),start_indices,chunks):
                        if pop_j == self_pop_j: # self-interactions
                            thread_num_this_pop_self = thread_num_this_pop_pair
                        else:
                            thread_num_this_pop_self = -1 # nonsense value (unused)
                        thread_info = np.array([
                            thread_num, start_index, thread_num_this_pop_self, num_threads_this_pop_self,
                            pop_name_index == other_pop_name_index # same_pop
                        ])

                        threads += [
                            threading.Thread(
                                target = inner_func,
                                args = (
                                    (thread_info,) + tuple(chunk) + unchunkable_args
                                )
                            )
                        ]
                        thread_num += 1
                    pop_j += 1

            # run the threads
            for thread in threads:
                thread.start()

            # stop the threads
            for thread in threads:
                thread.join()
        return func_multi_threaded(pops_args)


def make_multithread_simple(func, chunkable_args, unchunkable_args):
    """ Shortcut to multithreading for a single population with automatic chunking """

    return make_multithread(
        func,
        [[[
            [],
            chunkable_args,
            (0,0,0,1,1) + unchunkable_args
        ]]]
    )

def SRK2_force_calc(pops, status, sim):
    """
    Stochastic Runge-Kutta 2nd-order update following
    Branka and Heyes, Physical Review E, 60(2):2381, 1999.
    """

    if sim['save_state_every'] > 0:
        max_dt_usable = status.next_save_time - status.t
    else:
        max_dt_usable = -1 # nonsense value to ignore
    steps_until_update_grid_list = (config.n_steps_update_grid_list -
        status.update_grid_list_counter)
    time_vals = np.array([status.dt, status.dtmax,
        config.max_dist_per_step, config.RK_cheat_factor, max_dt_usable, steps_until_update_grid_list, config.n_steps_update_grid_list, status.next_output_step - status.stepnum
    ])

    pop_consts_list = [np.array([pop.D, pop.Dr]) for pop in pops]
    for pop_i, pop in zip(range(len(pops)),pops):
        if isinstance(pop, populations.WCA_Population):
            pop_consts = pop_consts_list[pop_i]
            pop_consts = np.append(
                pop_consts, np.array([pop.repel_d_max])
            )
            if isinstance(pop, populations.Chain_Collection):
                pop_consts = np.append(pop_consts,
                    np.array([ pop.rigid_rod_projection_factor,
                        pop.n_nearby_beads_dont_repel,
                        pop.longest_chain_length
                    ])
                )
            pop_consts_list[pop_i] = pop_consts

    pop_pairs = [
        [pop, pop_consts, [
            other_pop for other_pop in pops
            if ('forces' in dir(pop) or other_pop is pop)
        ]] for pop, pop_consts in zip(pops, pop_consts_list)
        if not pop.fixed
    ]

    num_pop_pairs = sum([len(item[2]) for item in pop_pairs])
    if config.n_threads < num_pop_pairs:
        config.n_threads = num_pop_pairs
        print(f'--Increasing n_threads to {config.n_threads}')
    max_n_threads = max(num_pop_pairs, sim['n_threads'])
    if config.n_threads > max_n_threads:
        config.n_threads = max_n_threads
        print(f'--Decreasing n_threads to {config.n_threads}')
    n_threads_per_pop_pair = max(1,int(config.n_threads // num_pop_pairs))
    n_threads_first_pop_pair = (
        config.n_threads - (num_pop_pairs - 1) * n_threads_per_pop_pair
    )
    shared_arr = np.zeros((config.n_threads, 6))
    shared_arr[:,0] = status.dt

    make_multithread(
        SRK.SRK_jit,
        [
            [
                [
                    chunk_chain(
                        (
                            n_threads_first_pop_pair
                            if (other_pop is pop and pop is pop_pairs[0][0])
                            else n_threads_per_pop_pair
                        ),
                        pop.total_pop, pop.chain_id
                    ) if 'chain_id' in dir(pop) else n_threads_per_pop_pair,
                    ( # chunked args
                        pop.pos, pop.pos1, pop.noise_vec, pop.where_active,
                        if_it_has(pop, 'tangent_list'),
                        if_it_has(pop, 'forces'),
                        if_it_has(pop, 'forces_aux'),
                        if_it_has(
                            pop, 'nbor_list',
                            otherwise = np.full(
                                (len(config.pop_name_list),0,0),0
                            )
                        )[
                            config.pop_name_list.index(other_pop.name)
                        ],
                    ),
                    ( # unchunked args
                        np.array([
                            config.pop_name_list.index(pop.name),
                            config.pop_name_list.index(other_pop.name),
                            len(pops)
                        ]),
                        config.Lx, config.grid_factor, config.SRK_order, status.b_calc_active_forces, time_vals, shared_arr,
                        # "full" arrays for this pop
                        if_it_has(
                            pop, 'chain_id', otherwise = np.full(0,0)
                        ),
                        pop.where_is_bead, pop.grid_list,
                        # "full" arrays of other_pop
                        other_pop.pos,
                        (
                            other_pop.pos1 if not other_pop.fixed
                            else other_pop.pos
                        ),
                        if_it_has(other_pop, 'tangent_list'),
                        other_pop.where_active, other_pop.grid_list,
                        other_pop.where_is_bead,
                        # constants pertaining to pop
                        if_it_has(
                            pop, 'rest_lengths', otherwise = np.zeros((0))
                        ),
                        if_it_has(
                            pop, 'phonebook', otherwise = np.full((0,0),0)
                        ),
                        pop_consts
                    ) + (interpop_forces(pop, other_pop)) # forces to calculate
                ] for other_pop in other_pops
            ] for pop, pop_consts, other_pops in pop_pairs
        ]
    )

    [status.dt, status.dt_used, steps_done, status.max_forces_strength,
        _, failed_grid_list_flag] = shared_arr[0]
    status.stepnum += int(steps_done)
    status.t += status.dt_used
    # print('---',status.t, status.dt_used)
    status.update_grid_list_counter = max(
        1, (int(steps_done) + status.update_grid_list_counter) % config.n_steps_update_grid_list
    )
    if failed_grid_list_flag < 0:
        for pop in pops:
            pop.update_grid_list(overwrite = True)
            if 'nbor_list' in dir(pop):
                if failed_grid_list_flag == -2:
                    pop.nbor_list = np.concatenate(
                        (pop.nbor_list, np.zeros_like(pop.nbor_list)), axis=2)
                for pop2 in pops:
                    pop.update_nbor_list_with_pop2(
                        pop2, grid_window = sim['grid_window']
                    )


def pop_index_diff(pop1, pop2):
    return ( config.pop_name_list.index(pop2.name) - config.pop_name_list.index(pop1.name)) % len(config.pop_name_list)


def interpop_forces(pop1, pop2):

    repulsive_only = pop1.b_init_growing or pop2.b_init_growing
    if pop2.name in pop1.special_interactions.keys():
        forces_list = pop1.special_interactions[pop2.name]
    elif pop1 is pop2:
        forces_list = pop1.self_interactions()
    else:
        forces_list = pop1.interactions(pop2)


    try:
        num_forces = len(forces_list)
    except:
        num_forces = 0

    if num_forces > 0:
        names_list = np.array([item[0] for item in forces_list])
        params_list = [item[1] for item in forces_list]
        params_array = np.zeros((
            len(params_list),
            max([len(item) for item in params_list])
        ))
        for i, params in zip(range(len(params_list)), params_list):
            for j, param in zip(range(len(params)), params):
                params_array[i,j] = param
    else:
        names_list = np.array(["none"])
        params_array = np.zeros((1,0))

    if repulsive_only:
        for i in range(len(names_list)):
            if names_list[i] in ['WCA', 'active_sliding_and_WCA']:
                params = params_array[i]
                if params[0] != 0:
                    # set LJ_cutoff_dist = repel_d_max
                    params[3] = (params[1]/params[0])**(1/6)

    return (names_list, params_array)


def if_it_has(pop, attr_str, otherwise = np.zeros((0,config.dim))):
    try:
        rtn = pop.__getattribute__(attr_str)
    except:
        rtn = otherwise
    return rtn


def chunk_chain(n_threads, total_pop, chain_id):
    chunklen = int(np.ceil(total_pop / n_threads))
    first_bead_in_chunk = 0
    split_indices = np.empty((n_threads-1),dtype=np.int)
    sii = 0
    while first_bead_in_chunk < total_pop and sii < n_threads - 1:
        if first_bead_in_chunk > 0:
            split_indices[sii] = first_bead_in_chunk
            sii += 1
        test_bead = min(
            max(first_bead_in_chunk,
                (sii+1) * chunklen - 1),
            total_pop - 1
        )
        last_ci = chain_id[test_bead]

        while test_bead < total_pop:
            if chain_id[test_bead] != last_ci:
                break
            test_bead += 1
        first_bead_in_chunk = test_bead
    return split_indices


@jit(**config.std_numba_kwargs())
def project_updates_for_rigid_rod(
    thread_info,
    rtn, prev_pos, total_forces, noise_vec,
    chains_phonebook, chain_id,
    rest_lengths, projection_factor, Lx, dt, diffusion_const, radius):
    """ For a given set of updates (i.e. forces + noise), calculates the
         resulting translation and rotation of a rigid rod experiencing the same
        forces + noise at the same points along its length. The modified update
        additionally removes displacements found in the current positions away
        from a perfect rigid rod configuration with given length.
    """
    thread_num, start_index = thread_info[:2]
    chain_length = chains_phonebook[0,0]
    COM_pos = np.empty(config.dim)
    mean_translation = np.empty(config.dim)
    r_hat = np.empty(config.dim)
    torque = np.empty(config.dim)

    num_chains = chains_phonebook.shape[0]
    half_Lx = 0.5 * Lx
    b_3D = config.dim == 3

    dt_over_r = dt / radius
    sqrt_2Ddt = (2 * diffusion_const * dt)**0.5 # sqrt(2) factor is already in noise_vec

    ci_min = chain_id[start_index]
    ci_max = chain_id[len(prev_pos) - 1 + start_index]
    prev_rest_length = 0
    prev_chain_length = 0

    for ci in range(ci_min, ci_max+1):
        chain_length = chains_phonebook[ci,0]
        rest_length = rest_lengths[ci]
        # check whether we can reuse some info from previous chain
        if chain_length!=prev_chain_length:
            out_of_line = np.empty((chain_length,config.dim))
            dist_from_COM = np.empty((chain_length))
            bi_this_chain = np.empty_like(chains_phonebook[ci,1:1+chain_length])
            updates = np.empty_like(out_of_line)
        if rest_length!=prev_rest_length or chain_length!=prev_chain_length:
            half_clm1 = (chain_length-1)/2
            for bii in range(chain_length):
                # bead's ideal signed distance from COM
                dist_from_COM[bii] = rest_length * (bii-half_clm1)
            moment_of_inertia = 0
            for bii in range(chain_length):
                for i in range(config.dim):
                    moment_of_inertia += dist_from_COM[bii]**2 # I = mr^2
        prev_chain_length = chain_length
        prev_rest_length = rest_length

        # indices of beads
        for bii in range(chain_length):
            bi_this_chain[bii] = chains_phonebook[ci,1+bii] - start_index
        # positions before projection
        prev_pos_this_chain = prev_pos[bi_this_chain]

        do_noise = (sqrt_2Ddt != 0)
        for bii in range(chain_length):
            bi = bi_this_chain[bii]
            for i in range(config.dim):
                updates[bii,i] = dt_over_r * total_forces[bi,i]
                if do_noise:
                     updates[bii,i] += sqrt_2Ddt * noise_vec[bi,i]

        # translation term
        for i in range(config.dim):
            mean_translation[i] = 0
        for bii in range(chain_length):
            for i in range(config.dim):
                mean_translation[i] += updates[bii,i]
        for i in range(config.dim):
            mean_translation[i] /= chain_length

        # calculate COM displacement from head bead,
        # allowing for periodic boundaries:
        for i in range(config.dim):
            COM_pos[i] = 0.
        r_mag = 0
        for bii in range(chain_length):
            for i in range(config.dim):
                sep = prev_pos_this_chain[bii,i] - prev_pos_this_chain[0,i]
                if sep < -half_Lx:
                    sep += Lx
                elif sep > half_Lx:
                    sep -= Lx
                COM_pos[i] += sep
                # use first and last beads for orientation unit vector
                if bii == chain_length-1:
                    r_hat[i] = sep
                    r_mag += sep**2
        # calculate COM position using head bead as reference
        for i in range(config.dim):
            COM_pos[i] = (prev_pos_this_chain[0,i] + COM_pos[i]/chain_length)%Lx
        # normalize orientation unit vector
        if r_mag > 0:
            r_mag = np.sqrt(r_mag)
            for i in range(config.dim):
                r_hat[i] /= r_mag

        # calculate deviations from perfect linearity currently
        for bii in range(chain_length):
            for i in range(config.dim):
                sep = prev_pos_this_chain[bii,i] - (
                    COM_pos[i] + r_hat[i]*dist_from_COM[bii]
                )
                if sep < -half_Lx:
                    sep += Lx
                elif sep >= half_Lx:
                    sep -= Lx
                out_of_line[bii,i] = sep

        # torque = r cross F, about COM:
        if not b_3D:
            for i in range(config.dim):
                torque[i] = 0
            for bii in range(chain_length):
                for i in range(config.dim):
                     torque[i] += ((2*i-1)*r_hat[1-i] # theta_hat = [-r_y, r_x]
                        * updates[bii,i]*dist_from_COM[bii]
                    )
            if moment_of_inertia != 0:
                for i in range(config.dim):
                    torque[i] /= moment_of_inertia

            for bii in range(chain_length):
                bi = bi_this_chain[bii]
                for i in range(config.dim):
                    # projected forces
                    updates[bii,i] *= 1.-projection_factor
                    updates[bii,i] += projection_factor * (
                        -out_of_line[bii,i]
                        + mean_translation[i]
                        + (torque[i] * dist_from_COM[bii]
                           * (2*i-1)*r_hat[1-i] # theta_hat
                        )
                    )
                    # new positions with this update
                    rtn[bi,i] = (prev_pos[bi,i]+updates[bii,i]) % Lx

        else:
            torque[:] = r_mag * np.sum(np.cross(r_hat, updates), axis=0)
            # projected forces
            updates *= 1.-projection_factor
            updates += projection_factor*(
                -out_of_line + mean_translation
                + (r_mag/moment_of_inertia) * np.cross(torque, r_hat)
            )
            # new positions with this update
            rtn[bi_this_chain,:] = np.mod(
                prev_pos_this_chain+updates, Lx
            )


@jit(**config.std_numba_kwargs())
def update_grid_list_mini(pos, grid_list, where_is_bead, system_size, start_gx, end_gx):
    """ update grid_list based on updated where_is_bead """

    grid_list_depth_less_one = grid_list.shape[-1] - 2
    for gx in range(start_gx, end_gx):
        for gy in range(grid_list.shape[1]):
            for gz in range(grid_list.shape[2]):
                grid_list[gx,gy,gz,0] = 0

    for bi in range(len(pos)):
        if start_gx <= where_is_bead[bi,0] < end_gx:
            grid_list_here = grid_list[
                where_is_bead[bi,0],where_is_bead[bi,1],where_is_bead[bi,2]
            ]
            grid_list_here[0] += 1
            if grid_list_here[0] < grid_list_depth_less_one:
                grid_list_here[grid_list_here[0]] = bi
            else:
                return False
    return True


@jit(**config.std_numba_kwargs())
def update_grid_list(
    pos, grid_list, where_is_bead, system_size, append_mode = False, overwrite = False, start = 0):
    """ Calculate which grid cell each bead is now in, updating both
        where_is_bead and grid_list
    """

    n_grid = grid_list.shape[1]
    n_grid_z = grid_list.shape[2]
    n_things = pos.shape[0]
    grid_spacing = system_size / n_grid
    for bi in range(n_things):
        for i in range(config.dim):
            where_is_bead[bi,i] = int(pos[bi,i] // grid_spacing)
        if config.dim == 2:
            where_is_bead[bi,2] = 0

    if append_mode or overwrite:
        for bi in range(n_things):
            where_is_bead[bi,0] = - where_is_bead[bi,0] - 1
        if overwrite:
            grid_list[:,:,:,0] = 0
    else:
        start = 0
        gxyz = np.full(3,0)
        for gxyz[0] in range(n_grid):
            for gxyz[1] in range(n_grid):
                for gxyz[2] in range(n_grid_z):
                    grid_list_here = grid_list[gxyz[0], gxyz[1], gxyz[2]]
                    num_here_now = 0
                    for bi in grid_list_here[1:1+grid_list_here[0]]:
                        moved = False
                        for i in range(config.dim):
                            if where_is_bead[bi,i] != gxyz[i]:
                                moved = True
                                break
                        if moved:
                            # if not np.sum(where_is_bead[bi] != gxyz):
                                grid_list_here[1+num_here_now] = bi
                                num_here_now += 1
                        else:
                            where_is_bead[bi,0] = - where_is_bead[bi,0] - 1
                    grid_list_here[0] = num_here_now
    # updated array starts out with only the beads that didn't change grid cells

    # place the beads that changed grid cells into their new slots in the array
    grid_list_depth = grid_list.shape[-1]
    for bi in range(n_things):
        if where_is_bead[bi,0] < 0:
            where_is_bead[bi,0] = - where_is_bead[bi,0] - 1
            grid_list_here = grid_list[
                where_is_bead[bi,0],where_is_bead[bi,1],where_is_bead[bi,2]
            ]
            if grid_list_here[0] < grid_list_depth - 2:
                grid_list_here[1+grid_list_here[0]] = start + bi
            else:
                return False
            grid_list_here[0] += 1
    return True


@jit(**config.std_numba_kwargs())
def update_nbor_list(
    thread_info,
    # chunked
    pos1, where_is_bead1, nbor_list1_pop2,
    # unchunked
    chain_id1, pos2, grid_list2, n_nearby_beads_dont_repel1, Lx,
    # optional
    grid_window = 1
    ):

    """ Update neighbor lists after each grid_list update """

    thread_num, start_index = thread_info[:2]
    b_3D = config.dim == 3
    same_pop = thread_info[4]

    n_grid = grid_list2.shape[0]
    max_dist_sq = (grid_window * Lx / n_grid)**2
    half_Lx = Lx/2
    nbor_list1_pop2[:,0] = 0 # wipe headcounts

    dg_start = - min(grid_window, int(n_grid // 2))
    dg_end = min(1+grid_window, n_grid + dg_start)
    PBC_flags_b1 = np.empty(config.dim)
    for b1 in range(where_is_bead1.shape[0]):
        if same_pop:
            b1_unchunked = b1 + start_index
        for i in range(config.dim):
            PBC_flags_b1[i] = (
                (where_is_bead1[b1,i] < grid_window + 1)
                or (where_is_bead1[b1,i] >= n_grid - (grid_window + 1))
            )

        gy2_start = dg_start + where_is_bead1[b1,1]
        gy2_end = where_is_bead1[b1,1] + dg_end

        if b_3D:
            gz2_start = dg_start + where_is_bead1[b1,2]
            gz2_end = where_is_bead1[b1,2] + dg_end
        else:
            gz2_start = 0
            gz2_end = 1
        for gx2 in range(
            dg_start + where_is_bead1[b1,0],
            where_is_bead1[b1,0] + dg_end
        ):
            if PBC_flags_b1[0]:
                gx2 %= n_grid
            for gy2 in range(gy2_start, gy2_end):
                if PBC_flags_b1[1]:
                    gy2 %= n_grid
                for gz2 in range(gz2_start, gz2_end):
                    if b_3D:
                        if PBC_flags_b1[2]:
                            gz2 %= n_grid
                    for b2 in grid_list2[
                        gx2,gy2,gz2,1:1+grid_list2[gx2,gy2,gz2,0]
                    ]:
                        if same_pop:
                            if chain_id1[b1_unchunked] == chain_id1[b2]:
                                if n_nearby_beads_dont_repel1 < 0:
                                    continue
                                if abs(b1_unchunked - b2) <= n_nearby_beads_dont_repel1:
                                    continue
                        dist_sq = 0
                        for i in range(config.dim):
                            sep = pos2[b2,i] - pos1[b1,i]
                            if PBC_flags_b1[i]:
                                if sep < - half_Lx:
                                    sep += Lx
                                elif sep >= half_Lx:
                                    sep -= Lx
                            dist_sq += sep**2

                        if dist_sq <= max_dist_sq:
                            if nbor_list1_pop2[b1,0] + 1 >= nbor_list1_pop2.shape[1]:
                                # nonsense value as error flag
                                nbor_list1_pop2[b1,0] = -1
                                return False
                            else:
                                nbor_list1_pop2[b1,0] += 1
                                nbor_list1_pop2[b1,nbor_list1_pop2[b1,0]] = b2
    return True


@jit(**config.std_numba_kwargs())
def update_nbor_list_few(
    thread_info,
    # chunked
    pos1, where_is_bead1, nbor_list1_pop2,
    # unchunked
    chain_id1, pos2, where_is_bead2, n_nearby_beads_dont_repel1, Lx, n_grid,
    # optional
    grid_window = 1
):

    thread_num, start_index = thread_info[:2]
    b_3D = config.dim == 3
    same_pop = thread_info[4]

    grid_spacing = Lx / n_grid
    max_dist_sq = (grid_window * Lx / n_grid)**2
    half_Lx = Lx/2
    nbor_list1_pop2[:,0] = 0 # wipe headcounts
    half_n_grid = n_grid/2

    for b1 in range(where_is_bead1.shape[0]):
        for b2 in range(len(pos2)):
            use_bead = True
            for i in range(config.dim):
                dg = where_is_bead2[b2,i] - where_is_bead1[b1,i]
                if dg < - half_n_grid:
                    dg += n_grid
                elif dg >= half_n_grid:
                    dg -= n_grid
                if np.abs(dg) > grid_window:
                    use_bead = False
                    break # don't have to check this bead
            if use_bead:
                if same_pop:
                    b1_unchunked = b1 + start_index
                    if chain_id1[b1_unchunked] == chain_id1[b2]:
                        if n_nearby_beads_dont_repel1 < 0:
                            continue
                        if abs(b1_unchunked - b2) <= n_nearby_beads_dont_repel1:
                            continue
                dist_sq = 0
                for i in range(config.dim):
                    sep = pos2[b2,i] - pos1[b1,i]
                    if sep < - half_Lx:
                        sep += Lx
                    elif sep >= half_Lx:
                        sep -= Lx
                    dist_sq += sep**2

                if dist_sq <= max_dist_sq:
                    if (
                        nbor_list1_pop2[b1,0] + 1
                        >= nbor_list1_pop2.shape[1]
                    ):
                        # nonsense value as error flag
                        nbor_list1_pop2[b1,0] = -1
                        return False
                    else:
                        nbor_list1_pop2[b1,0] += 1
                        nbor_list1_pop2[b1,nbor_list1_pop2[b1,0]] = b2

    return True

#!/usr/bin/env python3

import numpy as np
from numba import jit
import populations, config, forces, updates, force_rules
import threading
import math

## LLVM debug mode:
# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')

@jit(**config.std_numba_kwargs())
def SRK_jit(
    thread_info,
    ## chunked
    pos, pos1, noise_vec, where_active,
    tangent_list, total_forces, total_forces_aux, nbor_list,
    ## unchunked
    pops_info, Lx, grid_factor, SRK_order, b_calc_active_forces, time_vals, shared_arr,
    # "full" arrays for this pop
    chain_id_full, where_is_bead_full, grid_list,
    # "full" arrays of other_pop
    other_pop_pos_full, other_pop_pos1_full, other_pop_tangent_list_full, other_pop_where_active, other_pop_grid_list, other_pop_where_is_bead,
    # constants pertaining to pop
    rest_lengths, phonebook, pop_consts,
    # forces to calculate
    force_names, force_params
):
    ## unpack

    [dt, dtmax, max_dist_per_step, RK_cheat_factor, max_dt_usable,
        steps_until_update_grid_list, n_steps_update_grid_list, max_steps] = time_vals

    [thread_num, start_index, thread_num_this_pop_self,
        num_threads_this_pop_self, same_pop] = thread_info

    end_index = start_index + len(pos)
    chain_id = chain_id_full[start_index:end_index]
    where_is_bead = where_is_bead_full[start_index:end_index]

    [pop_index, other_pop_index, num_pops] = pops_info
    # difference in indices of pop and other_pop, and whether they are same
    pop_index_diff = (other_pop_index - pop_index) % num_pops

    diffusion_const = pop_consts[0]
    rot_diffusion_const = pop_consts[1]
    if len(pop_consts) > 2:
        radius = pop_consts[2]
        if len(pop_consts) > 3:
            [rigid_rod_projection_factor, n_nearby_beads_dont_repel, longest_chain_length] = pop_consts[3:]

    n_grid = grid_list.shape[1]

    b_calc_springs = b_calc_WCA = b_self_propel = b_active_sliding_and_WCA = b_Efield = b_motor_push = b_diffuse = b_walls = b_qq = b_get_orientations = False
    nbor_length_scale = 0.

    for fi in range(len(force_names)):
        force_name = force_names[fi]
        if force_name == "springs" and longest_chain_length > 1:
            b_calc_springs = True
            spring_params = force_params[fi,:6]
            b_get_orientations = (spring_params[4] != 0)
        elif force_name == "WCA":
            b_calc_WCA = True
            WCA_params = force_params[fi,:4]
            nbor_length_scale = max(nbor_length_scale, WCA_params[3])
        elif force_name == "self-propulsion" and b_calc_active_forces:
            b_self_propel = True
            active_params = force_params[fi,:3]
            v0 = active_params[0]
            v_tan_angle = active_params[2]
        elif force_name == "active_sliding_and_WCA":
            if b_calc_active_forces:
                b_active_sliding_and_WCA = True
                WCA_params = force_params[fi,:6]
            else:
                b_calc_WCA = True
                WCA_params = force_params[fi,:4]
            nbor_length_scale = max(nbor_length_scale, WCA_params[3])
        elif force_name == "Efield":
            b_Efield = True
            Efield_arr = force_params[fi,:config.dim]
        elif force_name == "motor_push" and b_calc_active_forces:
            b_motor_push = True
            motor_params = force_params[fi,:5]
            nbor_length_scale = max(nbor_length_scale, motor_params[0])
        elif force_name == "diffuse":
            b_diffuse = True
            b_motors_dont_diffuse_while_pushing = (force_params[fi,0] != 0)
        elif force_name == "walls":
            b_walls = True
            wall_params = force_params[fi]
        elif force_name == "quadrupole_quadrupole":
            b_qq = True
            qq_params = force_params[fi,:2]
            nbor_length_scale = max(nbor_length_scale, qq_params[1])

    grid_spacing = Lx / grid_list.shape[1]

    grid_window = max(1, int(np.ceil(grid_factor*
        nbor_length_scale / grid_spacing))
    )

    # preparing chunking of grid_list among only threads calculating position updates
    if same_pop:
        grid_list_chunk_size = int(
            np.ceil(len(grid_list) / num_threads_this_pop_self)
        )
        start_gx = thread_num_this_pop_self * grid_list_chunk_size
        end_gx = min(
            grid_list.shape[0],
            (thread_num_this_pop_self+1) * grid_list_chunk_size
        )

        total_forces_lcl = total_forces
        total_forces_aux_lcl = total_forces_aux

    # if not same_pop, thread writes forces to a different array than that used by same_pop threads. "Total_forces" pointer points to a local array instead while the pointer to the shared array is given a different name.
    else:
        total_forces_lcl = np.zeros_like(total_forces)
        total_forces_aux_lcl = np.zeros_like(total_forces_aux)

    shared_arr[thread_num,4] = 0 # counter for thread lock

    # loop until reaching step or time when required to return
    # or error flag is raised
    while ((shared_arr[thread_num, 2] < max_steps or max_steps < 0) and
        (shared_arr[thread_num, 1] < max_dt_usable or max_dt_usable < 0)
        and shared_arr[thread_num,5] >= 0):

        where_active[:] = -1 # wipe array, fill with nonsense value
        # need new noise_vec each iteration (one per population)
        if same_pop and diffusion_const > 0:
            for b1 in range(len(noise_vec)):
                if diffusion_const > 0:
                    for i in range(config.dim):
                        noise_vec[b1,i] = np.random.standard_normal()
        if same_pop and rot_diffusion_const > 0:
            for b1 in range(len(noise_vec)):
                for i in range(config.dim,2*config.dim):
                    noise_vec[b1,i] = np.random.standard_normal()

        if SRK_order == 4:
            # wipe array
            for b1 in range(len(total_forces_aux_lcl)):
                for i in range(config.dim):
                    total_forces_aux_lcl[b1,i] = 0
            force_prefactor = 1. # coefficient of force in virtual updates
            write_to_forces = total_forces_aux_lcl # for virtual update forces
            if not same_pop:
                self_write_to_forces = total_forces_aux
        elif SRK_order <= 2:
            # wipe array
            for b1 in range(len(total_forces_lcl)):
                for i in range(config.dim):
                    total_forces_lcl[b1,i] = 0
            write_to_forces = total_forces_lcl # for virtual and real update forces
            if not same_pop:
                self_write_to_forces = total_forces


        shared_arr[thread_num,5] = 0 # holds flag for grid list or nbor list requiring larger depth

        # loop over steps of SRK algorithm (2 iterations for SRK2, 4 for SRK4)
        for SRK_step in range(SRK_order):
            # a few step-dependent definitions here, so the rest is invariant for all iterations
            if SRK_order == 2:
                if SRK_step == 0:
                    # 1st step: check force at last real positions,
                    # store virtual update in pos1.
                    write_to_pos = pos1
                    write_to_pos_full = other_pop_pos1_full
                    force_prefactor = 1.
                    check_force_at_pos = pos
                    check_force_at_pos_full = other_pop_pos_full
                else:
                    # 2nd step: real update stored in pos, force calculated
                    # at virtual positions
                    write_to_pos = pos
                    write_to_pos_full = other_pop_pos_full
                    force_prefactor = 0.5 # turns sum of two force calculations into average
                    check_force_at_pos = pos1
                    check_force_at_pos_full = other_pop_pos1_full

            elif SRK_order == 4:

                if SRK_step <= 1:
                    dt_rescale_factor =  0.5
                    # to multiply dt by 1/2 in getting
                # 1st and 2nd virtual updates
                else:
                    dt_rescale_factor = 1 # for last virtual update and one real update
                if SRK_step == 0:
                    # use real old positions for first force calculation
                    check_force_at_pos = pos
                    check_force_at_pos_full = other_pop_pos_full
                else:
                    # force array is wiped before SRK loop but wiped again between iterations
                    for b1 in range(len(total_forces_aux_lcl)):
                        for i in range(config.dim):
                            total_forces_aux_lcl[b1,i] = 0
                 # evaluate force at positions of virtual updates from
                 # previous force calculation
                    check_force_at_pos = pos1
                    check_force_at_pos_full = other_pop_pos1_full

                if SRK_step == 3:
                    # last step is real update
                    write_to_pos = pos
                    write_to_pos_full = other_pop_pos_full
                else:
                    # other steps are virtual updates
                    write_to_pos = pos1
                    write_to_pos_full = other_pop_pos1_full

            else: # default is 1 (Euler)
                write_to_pos = pos
                write_to_pos_full = other_pop_pos_full
                force_prefactor = 1 # turns sum of two force calculations into average
                check_force_at_pos = pos
                check_force_at_pos_full = other_pop_pos_full

            thread_wait(shared_arr, thread_num)

            ### calculate the forces
            # springs
            if b_calc_springs:
                forces.calc_chain_spring_forces(
                    check_force_at_pos, tangent_list, chain_id,
                    write_to_forces, Lx, spring_params, rest_lengths,
                #    linear_spring_rule = force_rules.linear_spring_rule
                )
            thread_wait(shared_arr, thread_num) # wait for tangents to be updated

            if b_calc_WCA or b_active_sliding_and_WCA:
                # WCA
                forces.central_pair_force(
                    thread_info,
                    #chunked
                    check_force_at_pos, nbor_list, write_to_forces, tangent_list, where_active,
                    #unchunked
                    Lx, check_force_at_pos_full, WCA_params, other_pop_tangent_list_full, grid_window * Lx/n_grid,
                    #optional
                    b_calc_active_sliding = b_active_sliding_and_WCA,
                    force_rule = force_rules.central_pair_force_rule
                )

            if b_self_propel:
                forces.self_propel(
                    write_to_forces, tangent_list, #chunked
                    active_params #unchunked
                )

            if b_Efield:
                forces.Efield(
                    write_to_forces, tangent_list, chain_id, # chunked
                    Efield_arr #unchunked
                )

            if b_motor_push:
                forces.calc_motor_push(
                    # chunked
                    check_force_at_pos, nbor_list, write_to_forces, where_active, tangent_list,
                    #unchunked
                    Lx, check_force_at_pos_full, other_pop_where_active, motor_params
                )

            if b_walls:
                forces.walls(
                    pos, write_to_forces, where_is_bead,      # chunked
                    Lx, wall_params, n_grid,     # unchunked
                    grid_window = grid_window   # optional
                )

            if b_qq:
                forces.general_pair_force(
                    thread_info,
                    #chunked
                    check_force_at_pos, nbor_list, write_to_forces, tangent_list, where_active,
                    #unchunked
                    Lx, check_force_at_pos_full,
                    qq_params,
                    other_pop_tangent_list_full, grid_window * Lx/n_grid
                )

            # the not-same_pop threads take turns adding their results to the same_pop threads' force arrays
            for ii in range(1,num_pops):
                if pop_index_diff == ii:
                    for b1 in range(len(self_write_to_forces)):
                        for i in range(config.dim):
                            self_write_to_forces[b1,i] += write_to_forces[b1,i]
                            write_to_forces[b1,i] = 0
                thread_wait(shared_arr, thread_num)

            # For SRK4, accumulate intermediate force calculations in total_forces
            if SRK_order == 4:
                if same_pop:
                    if SRK_step == 0 or SRK_step == 3:
                        force_multiplier = 1./6.
                    else:
                        force_multiplier = 1./3.

                    if SRK_step == 0:
                        # need to overwrite values from last timestep
                        for b1 in range(len(total_forces)):
                            for i in range(config.dim):
                                total_forces[b1,i] = total_forces_aux[b1,i] * force_multiplier

                    # in subsequent iterations, accumulate in total_forces
                    else:
                        for b1 in range(len(total_forces)):
                            for i in range(config.dim):
                                total_forces[b1,i] += total_forces_aux[b1,i] * force_multiplier

                    # repurposing update
                    # (which uses "write_to_forces" pointer as forces)
                    # for real update
                    if SRK_step == 3:
                        write_to_forces = total_forces
                    thread_wait(shared_arr, thread_num)

            # on last iteration of SRK loop...
            if SRK_step == SRK_order - 1:
                # get max_forces_strength for this population:
                shared_arr[thread_num,3] = np.sqrt(2 * diffusion_const)
                if len(total_forces) > 0:
                    shared_arr[thread_num,3] += force_prefactor * np.max(np.sum(total_forces**2, axis=1))**0.5 / radius

                shared_arr[thread_num,2] += 1 # increment local step count
            thread_wait(shared_arr, thread_num)

            ### adapt timestep on last iteration of SRK loop
            if SRK_step == SRK_order - 1:

                # find max force over all threads
                max_forces_strength = np.max(shared_arr[:,3])

                if max_forces_strength == 0:
                    dt = dtmax
                else:
                    dt =  min(dtmax, max_dist_per_step / max_forces_strength)
                dt_used = RK_cheat_factor * dt

                if max_dt_usable > 0:
                    dt_used = min(dt_used, max_dt_usable - shared_arr[thread_num,1])

                shared_arr[thread_num,0] = dt # record most recent dt
                shared_arr[thread_num,1] += dt_used # track time elapsed within this SRK_fast call

                dt_for_update = dt_used # dt_used is applied in real update
            else:
                dt_for_update = dt # dt is applied for all virtual updates

            ### update (real or virtual)
            if same_pop:
                if SRK_order == 4:
                    dt_eff = dt_for_update * dt_rescale_factor * force_prefactor
                else:
                    dt_eff = dt_for_update * force_prefactor
                sqrt_2Ddt = (2 * dt_for_update * diffusion_const)**0.5

                if len(write_to_forces) > 0:
                    if rigid_rod_projection_factor != 0 and longest_chain_length > 2:
                        updates.project_updates_for_rigid_rod(
                            thread_info,
                            write_to_pos, # result array
                            check_force_at_pos, write_to_forces, noise_vec,
                            phonebook, chain_id_full, rest_lengths,
                            rigid_rod_projection_factor, Lx, dt_eff,
                            diffusion_const, radius
                        )
                    else:
                        # Division by repel_d_max institutes Stokes-Einstein relation for spheres:
                        # D ~ 1/r, so forces rescale as 1/r and noise rescales as sqrt(T/r), using diffusion_const  = T / r
                        dt_eff /= radius
                        for b1 in range(len(pos)):
                            for i in range(config.dim):
                                tmp = (
                                    pos[b1,i]
                                    + dt_eff * write_to_forces[b1,i]
                                )
                                if sqrt_2Ddt != 0:
                                    tmp += (
                                        sqrt_2Ddt * noise_vec[b1,i]
                                    )
                                tmp %= Lx
                                write_to_pos[b1,i] = tmp
                elif b_diffuse:
                    for b1 in range(len(pos)):
                        # by default, motors only diffuse if they aren't pushing
                        if where_active[b1] < 0 or not b_motors_dont_diffuse_while_pushing:
                            for i in range(config.dim):
                                write_to_pos[b1,i] = (
                                    pos[b1,i] + sqrt_2Ddt * noise_vec[b1,i]
                                ) % Lx

        # rotational diffusion (done AFTER all SRK steps, if at all)
        if rot_diffusion_const > 0 and not b_get_orientations:
            # prefactor, with factor of pi converting random translation to random angle change on unit circle/sphere
            sqrt_2Dr_dt = (
                2 * dt_for_update * rot_diffusion_const
            )**0.5 * np.pi
            for b1 in range(len(pos)):
                # project random vector into subspace perpendicular to tangent
                proj = 0.
                for i in range(config.dim):
                    proj += (
                        noise_vec[b1,config.dim+i]*tangent_list[b1,i]
                    )
                for i in range(config.dim):
                    noise_vec[b1,config.dim+i] -= proj*tangent_list[b1,i]
                # add noise vector to tangent and normalize
                tmp_mag = 0.
                for i in range(config.dim):
                    tmp = tangent_list[b1,i] + (
                        sqrt_2Dr_dt * noise_vec[b1,config.dim+i]
                    )
                    tmp_mag += tmp*tmp
                    tangent_list[b1,i] = tmp
                tmp_mag = math.sqrt(tmp_mag)
                for i in range(config.dim):
                    tangent_list[b1,i] /= tmp_mag

            # Note: updates (virtual and real) always add to pos which contains
            # positions of last real update. In real update, pos itself is updated, whereas pos1 stores positions of virtual updates.

        if shared_arr[thread_num,2] >= steps_until_update_grid_list:

            # update grid coordinates lookup
            if same_pop:
                for bi in range(len(where_is_bead)):
                    for i in range(config.dim):
                        where_is_bead[bi,i] = int(pos[bi,i] // grid_spacing)
            thread_wait(shared_arr, thread_num)

            # 1st iteration: upate grid_list (once per population)
            # 2nd iteration: update nbor_list (once per population pair)
            # Return error flag if depth of either array too small
            for ii in range(2):
                if ii == 0:
                    if same_pop and force_names[0] != 'none':
                        success = updates.update_grid_list_mini(
                            other_pop_pos_full, grid_list, where_is_bead_full, Lx, start_gx, end_gx
                        )
                    else:
                        success = True
                else:
                    if len(nbor_list) > 0:
                        if len(other_pop_pos_full) < n_grid**config.dim:
                            #If other_pop is not numerous, it's faster to cycle over other_pop outermost.
                            success = updates.update_nbor_list_few(
                                thread_info,
                                # chunked
                                pos, where_is_bead, nbor_list,
                                # unchunked
                                chain_id_full, other_pop_pos_full, other_pop_where_is_bead,  n_nearby_beads_dont_repel, Lx, n_grid,
                                # optional
                                grid_window = grid_window
                            )
                        else:
                            success = updates.update_nbor_list(
                                thread_info,
                                #chunked
                                pos, where_is_bead, nbor_list,
                                #unchunked
                                chain_id_full, other_pop_pos_full, other_pop_grid_list, n_nearby_beads_dont_repel, Lx,
                                #optional
                                grid_window = grid_window
                            )
                    else:
                        success = True

                if not success:
                    shared_arr[thread_num,5] = -ii
                thread_wait(shared_arr, thread_num)

                for thread_flag in shared_arr[:,5]:
                    if thread_flag < 0:
                        shared_arr[thread_num,5] = thread_flag
                        continue

            # next local step number at which to update grid + nbor lists:
            steps_until_update_grid_list += n_steps_update_grid_list


@jit(**config.std_numba_kwargs())
def thread_wait(shared_arr, thread_num):
    """ Rudimentary thread lock: Increments a counter in a shared array and waits until all other threads have reached that counter value. """

    if len(shared_arr) == 1:
        return
    shared_arr[thread_num,4] += 1 # increment counter
    while np.sum(shared_arr[:,4] < shared_arr[thread_num,4]):
        pass

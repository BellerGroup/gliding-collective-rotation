
#!/usr/bin/env python3

import numpy as np
from numba import jit
import populations
import config

"""
Functions to calculate forces individually or in common combinations
"""

@jit(**config.std_numba_kwargs())
def WCA(dist_sq, params):
    return (dist_sq**-4 * (params[0] - params[1]*dist_sq**-3))

@jit(**config.std_numba_kwargs())
def power_law(dist_sq, params):
    return ( dist_sq**(-1) if dist_sq > .1**2 else 0.)

@jit(**config.std_numba_kwargs())
def quadrupole_quadrupole(seps, dist_sq):

    if dist_sq != 0:
        x2 = seps[0]*seps[0]
        y2 = seps[1]*seps[1]
        z2 = seps[2]*seps[2]
        xy4 = x2*x2+y2*y2
        eight_z4 = 8*z2*z2
        y2z2 = y2*z2
        ds112 = dist_sq ** (11/2)
        xyterm = 45 * (xy4 - 12 * y2z2 + eight_z4 + 2 * x2 * (y2 - 6 * z2)) / ds112
        # repurpose seps as force vector
        seps[0] *= xyterm
        seps[1] *= xyterm
        seps[2] *= 15 * (
                15*xy4 - 40*y2z2 + eight_z4 + 10*x2*(3*y2 - 4*z2)
            ) / ds112
    else:
        seps[0] = 0.
        seps[1] = 0.
        seps[2] = 0.

@jit(**config.std_numba_kwargs())
def inbounds_diff(a,b,hb,bounds,out):
    ds = 0
    for i in range(len(a)):
        out[i] = a[i]-b[i]
        if out[i] < -hb:
            out[i] += bounds
        elif out[i] >= hb:
            out[i] -= bounds
        ds += out[i]**2
    return ds


@jit(**config.std_numba_kwargs())
def central_pair_force(
    thread_info,
    #chunked
    pos, nbor_list, forces, tangent_list, where_active,
    #unchunked
    Lx, pos2, force_params, tangent_list2, PBC_check_width,
    #optional
    b_calc_active_sliding = False,
    force_rule = WCA
):

    """ WCA/Lennard-Jones potential. For sliding active nematics, calculate neighbor-dependent active forces simultaeously with WCA.
    """

    start_idx = thread_info[1]
    end_idx = start_idx + len(pos)
    same_pop = thread_info[4]

    half_Lx = Lx/2
    LJ_cutoff_dist_sq = force_params[3]**2
    sq_least_dist_check = LJ_cutoff_dist_sq
    if b_calc_active_sliding:
        sliding_partner_maxdist_sq = force_params[5]**2
        sq_least_dist_check = min(
            sq_least_dist_check,sliding_partner_maxdist_sq
        )
    PBC_check_width_R = Lx - PBC_check_width

    seps = np.empty((config.dim))
    same_thread = False
    PBC_flags = np.full(config.dim,False)
    for b1 in range(pos.shape[0]):
        for i in range(config.dim):
            PBC_flags[i] = not (PBC_check_width <= pos[b1,i] < PBC_check_width_R)
        true_b1 = b1 + start_idx
        for j in range(nbor_list[b1,0]):
            b2 = nbor_list[b1,1+j]
            if same_pop:
                if true_b1 <= b2 < end_idx:
                    continue
                else:
                    same_thread = (true_b1 > b2 >= start_idx)
            ds = 0
            for i in range(config.dim):
                sep = pos2[b2,i] - pos[b1,i]
                if PBC_flags[i]:
                    if sep < -half_Lx:
                        sep += Lx
                    elif sep >= half_Lx:
                        sep -= Lx
                seps[i] = sep
                ds += sep**2
                # if ds has already exceeded square of smallest cutoff distance, don't bother checking other components
                if ds >= sq_least_dist_check:
                    continue

            if b_calc_active_sliding:
                if ds < sliding_partner_maxdist_sq:
                    if tangent_list[b1].dot(tangent_list2[b2]) < 0:
                        for i in range(config.dim):
                            forces[b1,i] += (force_params[4] # <- v0
                                * tangent_list[b1,i])
                        where_active[b1] += 1

            if 0 < ds <= LJ_cutoff_dist_sq:
                if force_params[2]!=0:
                    sep_mag = ds**0.5
                    # adjust factor of distance in seps to match new ds
                    # map values <0 -> 0 before squaring => not used
                    new_sep_mag = max(0,(sep_mag+force_params[2]))
                    rescale = new_sep_mag/sep_mag
                    for i in range(config.dim):
                        seps[i] *= rescale
                    ds = new_sep_mag**2

                coeff = force_rule(ds, force_params)
                if same_thread:
                    b2_this_thread = b2 - start_idx
                    for i in range(config.dim):
                        fcomp = seps[i] * coeff
                        # equal and opposite forces:
                        forces[b1,i] += fcomp
                        forces[b2_this_thread,i] -= fcomp
                else:
                    for i in range(config.dim):
                        forces[b1,i] += seps[i] * coeff



@jit(**config.std_numba_kwargs())
def general_pair_force(
    thread_info,
    #chunked
    pos, nbor_list, forces, tangent_list, where_active,
    #unchunked
    Lx, pos2, force_params, tangent_list2, PBC_check_width,
    #optional
    force_rule = quadrupole_quadrupole
):

    """ Pair force with general dependence on separation vector.
    """

    start_idx = thread_info[1]
    end_idx = start_idx + len(pos)
    same_pop = thread_info[4]

    half_Lx = Lx/2
    cutoff_dist_sq = force_params[1]**2
    PBC_check_width_R = Lx - PBC_check_width

    seps = np.empty((config.dim))
    same_thread = False
    PBC_flags = np.full(config.dim,False)
    for b1 in range(pos.shape[0]):
        for i in range(config.dim):
            PBC_flags[i] = not (PBC_check_width <= pos[b1,i] < PBC_check_width_R)
        true_b1 = b1 + start_idx
        for j in range(nbor_list[b1,0]): # nbor_list[:,0] is number of neighbors, nbor_list[b1,j] is (j-1)th nbor of b1 for 1 <= j < 1 + nbor_list[b1,0]
            b2 = nbor_list[b1,1+j] # index of 2nd bead
            if same_pop:
                if true_b1 <= b2 < end_idx: # use only b1 > b2
                    continue
                else:
                    same_thread = (true_b1 > b2 >= start_idx)
            ds = 0
            for i in range(config.dim): # x,y,z
                sep = pos2[b2,i] - pos[b1,i] # separation vector b1 -> b2
                if PBC_flags[i]:
                    if sep < -half_Lx:
                        sep += Lx
                    elif sep >= half_Lx:
                        sep -= Lx
                seps[i] = sep
                ds += sep**2 # distance squared
            if 0 < ds <= cutoff_dist_sq: # if within cutoff dist
                force_rule(seps, ds) # repurposes seps as force vector
                if same_thread:
                    b2_this_thread = b2 - start_idx
                    for i in range(config.dim):
                        seps[i] *= force_params[0]
                        forces[b1,i] += seps[i]
                        forces[b2_this_thread,i] -= seps[i]
                else:
                    for i in range(config.dim):
                        seps[i] *= force_params[0]
                        forces[b1,i] += seps[i]

@jit(**config.std_numba_kwargs())
def self_propel(
    forces, tangent_list, #chunked
    active_params #unchunked
):
    v0 = active_params[0]
    v_tan_angle = active_params[2]
    # no v_tan_angle or 3D:
    if v_tan_angle == 0. or config.dim != 2:
        for b1 in range(len(forces)):
            for i in range(config.dim):
                forces[b1,i] += v0 * tangent_list[b1,i]
    else: # v_tan_angle
        sn = np.sin(v_tan_angle)
        cs = np.cos(v_tan_angle)
        for b1 in range(len(forces)):
            forces[b1,0] += v0 * (cs*tangent_list[b1,0]-sn*tangent_list[b1,1])
            forces[b1,1] += v0 * (sn*tangent_list[b1,0]+cs*tangent_list[b1,1])


@jit(**config.std_numba_kwargs())
def Efield(
    forces, tangent_list, chain_id, # chunked
    Efield_arr
):
    bi_start = 0
    while bi_start < len(forces):
        bi_end = bi_start + 1
        ci = chain_id[bi_start]
        while bi_end < len(forces) and chain_id[bi_end] == ci:
            bi_end += 1
        for bi in range(bi_start, bi_end):
            order_on_chain = bi - bi_start
            #!! this is quantitatively wrong
            sgn = 2 * (
                (bi - bi_start) < (bi_end - bi_start) / 2 ) - 1
            dp = sgn * tangent_list[bi].dot(Efield_arr)
            for i in range(config.dim):
                forces[bi,i] += dp * Efield_arr[i]
        bi_start = bi_end


@jit(**config.std_numba_kwargs())
def calc_motor_push(
    # chunked
    pos_b, nbor_list_b, forces_b, where_active_b, tangent_list,
    #unchunked
    Lx, pos_m, where_active_m, force_params
):

    """ Calculate active force on filaments as determined by motors.
    """
    k_on = force_params[3]
    k_off = force_params[4]
    half_Lx = Lx/2
    #where_active_m[:] = -1 #! does this detach all motors from beads everytime?
    for bi in range(pos_b.shape[0]): #!over all the bead numbers
        seps = np.empty((config.dim))
        for mii in range(nbor_list_b[bi,0]): #!over number of motor-neighbors of a bead bi

            mi = nbor_list_b[bi,1+mii] #! nth neighbor-motor of bead bi
            if where_active_m[mi] >= 0:# motor already pushed
                if np.random.random() > k_off:
                    continue #motor stays attached and pushes
                else:
                    where_active_m[mi] = -1 #motor gets detached
                    where_active_b[bi] -= 1 #number of motors attached to bead bi decreases

            if where_active_m[mi] < 0:
                if np.random.random() < k_on:
                    ds = inbounds_diff(pos_m[mi],pos_b[bi],half_Lx,Lx,seps)
                    if ds <= force_params[0]:
                        where_active_b[bi] += 1
                        where_active_m[mi] = bi # assign bead index to this motor
                        for i in range(config.dim):
                            forces_b[bi,i] += force_params[2] * tangent_list[bi,i]
                        if force_params[1]:
                            continue # move on to next bead

@jit(**config.std_numba_kwargs())
def calc_chain_spring_forces(
    #chunked
    pos, tangent_list, chain_id, forces,
    # unchunked
    Lx, spring_params, rest_lengths
    ):
    """ Calculation of linear spring and bending spring forces.
    """

    half_Lx = Lx / 2
    b_calc_f_bend = (spring_params[1] != 0)
    max_b1 = len(pos) - 1
    b_get_orientations = (spring_params[4] != 0)

    tangents = np.empty((2,config.dim))
    new_chain = True

    bond_max_length_sq = 0. # used in linear spring for FENE only

    for b1 in range(len(pos)):
        if new_chain:
            ti = 0
            tj = 1
            ci = chain_id[b1]
            chain_a = rest_lengths[ci]
            if spring_params[3] == 1.:
                max_strentch_len = spring_params[5] * chain_a
                chain_a_6 = chain_a**6 # used as 2*sigma_6 in WCA

        if b1 == max_b1:
            end_of_chain = True
        elif chain_id[b1+1] != ci:
            end_of_chain = True
        else:
            end_of_chain = False
        if not end_of_chain:
            dist_sq = inbounds_diff(pos[b1],pos[b1+1],half_Lx,Lx,tangents[ti])

            # displacement vector pointing from bead behind -> bead b1
            # (unnormalized!)
            if dist_sq != 0:
                inv_dist = dist_sq**(-0.5)
                # LINEAR SPRING FORCE
                # note: factor of distance missing because we normalize
                # tangents AFTER multiplying them my this magnitude
                if spring_params[3]:
                    # FENE (bi-directional, with nonzero restlength)
                    # Note: Typical FENE has zero rest length and is added to a short-ranged WCA repulsion. We don't do that here.
                    stretch_length = dist_sq**0.5 - chain_a
                    f_spring_mag = (
                        -spring_params[0] * inv_dist * stretch_length * 1. / (
                            np.abs(stretch_length**2/max_strentch_len**2 - 1.)
                            if stretch_length**2 != max_strentch_len and max_strentch_len != 0.
                            else 0.
                        )
                    )
                else:
                    # Hookean
                    f_spring_mag = spring_params[0] * (chain_a * inv_dist - 1.)
                for i in range(config.dim):
                    f_spring_comp = tangents[ti,i] * f_spring_mag
                    forces[b1,i] += f_spring_comp
                    forces[b1+1,i] -= f_spring_comp
                    tangents[ti,i] *= inv_dist # normalize

        if b_get_orientations or b_calc_f_bend:
            if new_chain:
                if b_get_orientations:
                    for i in range(config.dim):
                        tangent_list[b1,i] = tangents[ti,i]
            elif end_of_chain:
                if b_get_orientations:
                    for i in range(config.dim):
                        tangent_list[b1,i] = tangents[tj,i]
            else: # BENDING SPRING ENERGY (only for interior beads)
                fwd_tngt_dot_bkwd_tngt = (tangents[ti]).dot(tangents[tj])
                if -1 < fwd_tngt_dot_bkwd_tngt < 1:
                    if b_calc_f_bend:
                        phi_term = np.arccos(fwd_tngt_dot_bkwd_tngt)
                        # spontaneous bond angle
                        if spring_params[2] != 0:
                            phi_term += spring_params[2] * np.sign(
                                - tangents[ti,0] * tangents[tj,1]
                                + tangents[ti,1] * tangents[tj,0]
                            )
                        phi_term *= spring_params[1] / np.sqrt(
                            1.-fwd_tngt_dot_bkwd_tngt**2
                        )
                        for i in range(config.dim):
                            f_bend_forward_here_comp = phi_term * (
                                fwd_tngt_dot_bkwd_tngt * tangents[tj,i] - tangents[ti,i]
                            )
                            f_bend_backward_here_comp = phi_term * (
                                tangents[tj,i] - fwd_tngt_dot_bkwd_tngt * tangents[ti,i]
                            )
                            forces[b1,i] += (
                                f_bend_forward_here_comp
                                + f_bend_backward_here_comp
                            )
                            forces[b1-1,i] -= f_bend_forward_here_comp
                            forces[b1+1,i] -= f_bend_backward_here_comp
                    if b_get_orientations:
                        coeff = ( 2 * (1 + fwd_tngt_dot_bkwd_tngt) )**-0.5
                        for i in range(config.dim):
                            tangent_list[b1,i] = coeff * (
                                tangents[ti,i] + tangents[tj,i]
                            )

                else:
                    # perfect alignment:
                    if b_get_orientations:
                        for i in range(config.dim):
                            tangent_list[b1,i] = tangents[ti,i] # ?? no force when we have a spontaneous bond angle?

        tj = ti
        ti = 1 - ti
        new_chain = end_of_chain


@jit(**config.std_numba_kwargs())
def walls(
    pos, forces, where_is_bead,  #chunked
    Lx, force_params, n_grid, #unchunked
    grid_window = 1,
    force_rule = WCA
):

    """ Walls in x, y, and/or z direction employing a central pair force rule.
    """

    half_Lx = Lx/2
    forbidden_zone_width = Lx/n_grid*grid_window/2
    for b1 in range(pos.shape[0]):
        for i in range(config.dim):
            if force_params[-3+i] != 0:
                if not 2*grid_window <= where_is_bead[b1,i] < n_grid - 2*grid_window:
                    sep = pos[b1,i]
                    if sep >= half_Lx:
                        sep -= (Lx-forbidden_zone_width)
                    else:
                        sep -= forbidden_zone_width
                    if force_params[2] != 0: # nonzero WCA_shift
                        # map values <0 -> 0 before squaring => not used
                        ds = (max(0, abs(sep)-force_params[2]))
                        sep = np.sign(sep)*ds
                        ds = ds**2
                    else:
                        ds = sep**2
                    if 0 < ds <= force_params[3]:
                        forces[b1,i] += -sep * force_rule(ds, force_params)

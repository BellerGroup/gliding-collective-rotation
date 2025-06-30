#!/usr/bin/env python3
import numpy as np
import updates, config
import math

"""
Class definitions for various "population" types including bead-spring chains
and motors.
"""


def unit_vector(angle_list):
    if angle_list.shape[0] == 1:
        return np.array([math.cos(angle_list[0]),math.sin(angle_list[0])])
    elif angle_list.shape[0] == 2:
        sin_theta = math.sin(angle_list[1])
        return np.array([sin_theta * math.cos(angle_list[0]),
                  sin_theta * math.sin(angle_list[0]),
                  math.cos(angle_list[1])
                ])


def force_params_from_rules(pop, force_name, replacements):
    """ Helper function for creating special interactions, directing parameters
        from the front-end file to the appropriate force calculation
    """
    if force_name == "WCA":
        return make_WCA_replacements(pop, replacements)
    else:
        print("Error: invalid force_name in populations.force_params_from_rules(): " + force_name)
        exit()


def make_WCA_replacements(pop, kwargs):
    """ Translation from optional replacement values for WCA force to the parameters sent to forces.py
    """
    WCA_params_in = [0,0,0,0]
    strings_in =  ['repel_d_max', 'WCA_epsilon', 'WCA_shift', 'LJ_cutoff_factor']
    pop_vals = [ pop.repel_d_max, pop.WCA_epsilon, pop.WCA_shift, pop.LJ_cutoff_factor ]
    for i in range(len(strings_in)):
        if strings_in[i] in kwargs.keys():
            WCA_params_in[i] = kwargs[strings_in[i]]
        else:
            WCA_params_in[i] = pop_vals[i]
    return make_WCA_params(*tuple(WCA_params_in))


def make_WCA_params(repel_d_max, WCA_epsilon, WCA_shift, LJ_cutoff_factor):
    """ Translation from command-line values for WCA force to the parameters
        sent to forces.py
    """
    LJ_cutoff_dist = LJ_cutoff_factor * repel_d_max - WCA_shift
    two_WCA_sigma_6 = repel_d_max**6 # -> 1
    WCA_factor = 12 * two_WCA_sigma_6 * WCA_epsilon # -> 12
    WCA_factor_times_two_sigma_6 = WCA_factor * two_WCA_sigma_6 # -> 12
    return np.array([
        WCA_factor, WCA_factor_times_two_sigma_6, WCA_shift, LJ_cutoff_dist
    ])

                            ### Classes ###
class Population:
    """ Base class """
    def __init__(self, total_pop, sim, name = "pop", announcements = True):
        self.total_pop = total_pop
        self.pos = np.empty((total_pop, config.dim), dtype=np.float)
        self.pos1 = np.empty_like(self.pos)
        self.noise_vec = np.empty((total_pop, 2*config.dim), dtype=np.float)
        self.name = name
        self.special_interactions = {}
        self.v0 = 0.

        self.where_active = np.full(total_pop, 0)
        self.n_nearby_beads_dont_repel = 0
        self.b_init_growing = False
        self.pos_init_noise = sim['pos_init_noise']
        self.D = sim['D']
        self.Dr = sim['Dr']
        self.initialized = False
        self.fixed = False
        self.fix_after_grow = False
        if config.b_announcements:
            print(f'--{self.name}: creating {self.total_pop} particles')
    # synonym
    def how_many(self):
        return self.total_pop

    def make_empty_grid_list(self, num_pops = 1):
        self.n_grid = config.n_grid
        if config.dim == 2:
            n_grid_z = 1
        else:
            n_grid_z = self.n_grid
        init_grid_list_depth = max(1,int((8 * self.total_pop / self.n_grid**(config.dim))))
        # Factor of 8 above is arbitrary, but updates.update_gridlist automatically
        # expands grid_list if density is higher than this choice allows.

        self.grid_list = np.zeros((self.n_grid, self.n_grid, n_grid_z,
            1 + init_grid_list_depth), dtype=np.int)

        self.where_is_bead = np.empty((self.total_pop,3),dtype=np.int)
        if config.dim == 3:
            self.where_is_bead[:,2] = 0

    def update_grid_list(self, append_mode = False, overwrite = False, start = 0):
            while not updates.update_grid_list(
                self.pos, self.grid_list, self.where_is_bead,
                config.Lx, append_mode = append_mode, overwrite = overwrite, start = start
            ):
                self.grid_list = np.concatenate(
                    (np.zeros_like(self.grid_list),)*2, axis=-1
                )
                overwrite = True

    def interactions(self, pop):
        return []

    def self_interactions(self):
        return []

    def add_special_interaction(self, pop_name, force_name, **replacements):
        if not pop_name in self.special_interactions.keys():
            self.special_interactions[pop_name] = []
        self.special_interactions[pop_name].append(
            [ force_name, force_params_from_rules(self, force_name, replacements) ]
        )

    def size_scale(self):
        return config.Lx / np.sqrt(self.total_pop)

    def random_initialization(self):
        self.pos[:] = np.mod(config.Lx / 2 * (1. + self.pos_init_noise * (2 * np.random.rand(self.total_pop,config.dim) - 1)), config.Lx)
        self.initialized = True

    def set_init_grow(self):
        self.b_init_growing = True

    def end_init_grow(self):
        self.b_init_growing = False
        if self.fix_after_grow:
            self.fixed = True


class Forces_Population(Population):
    """ Population that can experience forces """
    def __init__(self, total_pop, sim, name = "beads"):
        super().__init__(total_pop, sim, name = name)
        self.reset_force_arrays()

    def reset_force_arrays(self):
        self.forces = np.zeros((self.total_pop,config.dim))
        if config.SRK_order == 2:
            self.forces_aux = np.empty((self.total_pop,0))
        elif config.SRK_order == 4:
            self.forces_aux = np.empty_like(self.forces)

    def update_nbor_list_with_pop2(self, pop2, grid_window = 1):
        pop2_index_in_pop1 = config.pop_name_list.index(pop2.name)
        if pop2 is self:
            grid_window = max(
                1, int(np.ceil(self.size_scale() * config.n_grid / config.Lx))
            )
        success = False
        while not success:
            success = updates.update_nbor_list(
                np.array([0,0,0,1,(self is pop2)]),
                self.pos,
                self.where_is_bead,
                self.nbor_list[pop2_index_in_pop1],
                self.chain_id,
                pop2.pos,
                pop2.grid_list,
                self.n_nearby_beads_dont_repel,
                config.Lx,
                grid_window = grid_window
            )
            if np.all(self.nbor_list[pop2_index_in_pop1,:,0] >= 0):
                success = True
            else:
                self.nbor_list = np.concatenate(
                    (self.nbor_list, np.zeros_like(self.nbor_list)), axis=2)

    def make_empty_grid_list(self, num_pops = 1):
        super().make_empty_grid_list(num_pops = num_pops)
        self.nbor_list = np.zeros((num_pops, self.total_pop, 1 + 8), dtype=np.int)


class WCA_Population(Forces_Population):
    """ Beads with steric interactions """
    def __init__(self, total_pop, sim, name = "beads"):
        super().__init__(total_pop, sim, name = name)
        self.reset_WCA_consts(sim)
        self.set_size(self.repel_d_max, sim = sim)
        self.sim = sim

    def set_size(self, diam, sim = None):
        if sim == None:
            sim = self.sim
        self.repel_d_max = diam
        self.diam = diam
        if self.D < 0:
            self.D = sim['T'] / self.repel_d_max
        if self.Dr < 0:
            self.Dr = 3 * self.D / (4 * self.repel_d_max**2)

    def reset_WCA_consts(self,sim):
        self.repel_d_max = sim['repel_d_max']
        self.WCA_epsilon = sim['WCA_epsilon']
        self.WCA_shift = sim['WCA_shift']
        self.LJ_cutoff_factor = sim['LJ_cutoff_factor']
        self.adjust_WCA_params()
        super().reset_force_arrays()

    def adjust_WCA_params(self):
        self.WCA_params = make_WCA_params(self.repel_d_max, self.WCA_epsilon, self.WCA_shift, self.LJ_cutoff_factor)

    def size_scale(self):
        return self.repel_d_max

    def self_interactions(self):
        forces_list = super().self_interactions()
        forces_list.append(["WCA", self.WCA_params])
        if np.any(config.walls):
            forces_list.append(["walls", np.append(self.WCA_params, config.walls)])
        return forces_list

    def interactions(self, other_pop):
        forces_list = super().interactions(other_pop)
        if isinstance(other_pop, WCA_Population):
            forces_list += [
                ["WCA", 0.5 * (self.WCA_params + other_pop.WCA_params)]
            ]
        return forces_list

    def set_init_grow(self):
        super().set_init_grow()
        self.repel_d_max_save = self.repel_d_max

    def end_init_grow(self):
        super().end_init_grow()
        self.repel_d_max = self.repel_d_max_save
        self.adjust_WCA_params()


class Oriented_WCA_Population(WCA_Population):
    """ Beads with steric interactions, capable of being active along their
        orientation directions
    """
    def __init__(self, total_pop, sim, name = "beads"):
        super().__init__(total_pop, sim, name = name)
        self.reset_v0(sim)
        self.tangent_list = np.empty((self.total_pop, config.dim))
        self.orientations = np.empty((self.total_pop, config.dim - 1))
        self.orientations[:,0] = (
            sim['phi_init'] * np.ones((self.how_many()))
             + sim['phi_init_noise']
             * np.random.normal(size=(self.how_many()))
        )
        if config.dim == 3:
            self.orientations[:,1] = np.full((self.total_pop), 0.5 * np.pi)
        for bi in range(self.total_pop):
            self.tangent_list[bi] = unit_vector(self.orientations[bi])
        self.phi_init = sim['phi_init']
        self.phi_init_noise = sim['phi_init_noise']
        self.Efield = np.zeros(config.dim)

    def reset_v0(self,sim):
        self.v0 = sim['v0']
        self.where_active = np.full((self.total_pop),0)

    def self_interactions(self):
        forces_list = super().self_interactions()
        if self.Efield.dot(self.Efield) > 0:
            forces_list.append(["Efield", self.Efield])
        return forces_list

    def interactions(self, pop):
        forces_list = super().interactions(pop)
        if isinstance(pop, Motor_Population):
            forces_list += [
                [
                    "motor_push",
                    np.array([
                        pop.motor_max_dist_sq, float(pop.b_one_motor_per_bead), pop.imparts_force, pop.k_on, pop.k_off
                    ])
                ]
            ]
        return forces_list

    def random_initialization(self):
        self.pos[:] = np.mod(config.Lx / 2 * (1. + self.pos_init_noise * (2 * np.random.rand(self.total_pop, config.dim) - 1)), config.Lx)
        self.orientations[:,0] = np.mod(self.phi_init + 2 * np.pi * self.phi_init_noise * np.random.rand(self.total_pop) + np.pi, 2*np.pi) - np.pi
        self.initialized = True

    def set_init_grow(self):
        super().set_init_grow()
        self.v0_save = self.v0
        self.v0 = 0.

    def end_init_grow(self):
        super().end_init_grow()
        self.v0 = self.v0_save



class Chain_Collection(Oriented_WCA_Population):
    """ Particles from an Oriented_WCA_Popoulation are strung together into
        chains, possibly active, of possibly different lengths
    """
    def __init__(self, num_subpops, total_pop, sim, name = "beads", chain_id = [None]):
        super().__init__(total_pop, sim, name = name)
        self.num_subpops = num_subpops
        # by default, apportion beads equally among chains
        if chain_id[0] == None:
            self.chain_id = np.asarray(np.floor(np.arange(self.total_pop))/(self.total_pop / self.num_subpops),dtype=np.int)
        self.inherited_label = self.chain_id.copy()
        self.reset_constants(sim)
        self.make_phonebook()

    def reset_constants(self,sim):
        self.k = sim['k']
        self.reference_rest_length = sim['a']
        self.rest_lengths = np.full(self.num_subpops, sim['a'])
        self.a_init_factor = sim['a_init_factor']
        self.spontaneous_bond_angle = sim['spontaneous_bond_angle']
        if sim['kappa'] >= 0:
            self.rigid_rod_projection_factor = sim['rigid_rod_projection_factor']
            self.kappa = sim['kappa']
        else:
            self.rigid_rod_projection_factor = 1
            self.kappa = 0

        self.n_nearby_beads_dont_repel = sim['n_nearby_beads_dont_repel']
        self.v_tan_angle = sim['v_tan_angle']
        self.b_use_tangent_as_propulsion_direction = sim['b_use_tangent_as_propulsion_direction']
        self.sliding_partner_maxdist = sim['sliding_partner_maxdist']
        self.b_init_growing = False
        self.b_grow = sim['b_grow']
        self.grow_rate = sim['grow_rate']
        self.grow_rate_noise = sim['grow_rate_noise']
        self.b_FENE = sim['b_FENE']
        self.FENE_tolerance = sim['FENE_tolerance']
        self.reset_v0(sim)
        self.reset_WCA_consts(sim)

    def spring_params(self):
        return np.array([
            self.k,
            self.kappa,
            self.spontaneous_bond_angle,
            float(int(self.b_FENE)),
            float(self.b_get_orientations),
            self.FENE_tolerance
        ])

    def active_params(self):
        return np.array([
            self.v0, self.sliding_partner_maxdist, self.v_tan_angle
        ])

    def make_phonebook(self):
        self.longest_chain_length = np.max(np.bincount(self.chain_id))
        self.phonebook = np.full(
            (self.num_chains(),1+self.longest_chain_length),-1
        )
        self.phonebook[:,0] = 0
        for bi in range(self.total_pop):
            this_chain_id = self.chain_id[bi]
            self.phonebook[this_chain_id,1+self.phonebook[this_chain_id,0]] = bi
            self.phonebook[this_chain_id,0] += 1
        self.set_order_on_chain()

    # synonym
    def num_chains(self):
        return self.num_subpops

    def set_init_grow(self):
        super().set_init_grow()
        self.rigid_rod_projection_factor_save = self.rigid_rod_projection_factor
        self.n_nearby_beads_dont_repel_save = self.n_nearby_beads_dont_repel
        self.WCA_shift_save = self.WCA_shift
        self.D_save = self.D
        self.n_nearby_beads_dont_repel = -1
        self.kappa_save = self.kappa
        self.k_save = self.k
        self.kappa = 0
        self.k = 1
        self.WCA_shift = 0
        self.D = 0.


    def end_init_grow(self):
        super().end_init_grow()
        self.rigid_rod_projection_factor = self.rigid_rod_projection_factor_save
        self.n_nearby_beads_dont_repel = self.n_nearby_beads_dont_repel_save
        self.rest_lengths[:] = self.reference_rest_length
        self.kappa = self.kappa_save
        self.k = self.k_save
        self.WCA_shift = self.WCA_shift_save
        self.adjust_WCA_params()
        self.D = self.D_save

    def add_new_chain(self, num_new_beads, inherited_label = -1):
        new_chain_number = np.max(self.chain_id) + 1
        bi_new_chain = self.total_pop + np.arange(num_new_beads)
        self.chain_id = np.append(
            self.chain_id, np.full(num_new_beads,new_chain_number)
        )
        self.order_on_chain = np.append(
            self.order_on_chain, np.arange(num_new_beads)
        )
        self.total_pop += num_new_beads
        self.num_subpops += 1
        self.where_active = np.append(
            self.where_active, np.full(num_new_beads,0)
        )
        if inherited_label == -1:
            inherited_label = np.max(self.inherited_label) + 1
        self.inherited_label = np.append(
            self.inherited_label, np.full(num_new_beads, inherited_label)
        )
        self.pos = np.concatenate((
            self.pos, np.zeros((num_new_beads,config.dim))
        ),axis=0)
        self.pos1 = np.concatenate((
            self.pos1, np.zeros((num_new_beads,config.dim))
        ),axis=0)
        self.tangent_list = np.concatenate((
            self.tangent_list, np.zeros((num_new_beads, config.dim))
        ), axis=0)
        self.orientations = np.concatenate((
            self.orientations, np.zeros((num_new_beads, config.dim - 1))
        ), axis=0)
        self.phonebook = np.concatenate((
            self.phonebook, np.zeros_like(self.phonebook[0:1])
        ), axis=0)
        self.phonebook[new_chain_number][0] = num_new_beads
        self.phonebook[new_chain_number,1:1+num_new_beads] = bi_new_chain
        self.where_is_bead = np.concatenate((
            self.where_is_bead, np.empty(
                (num_new_beads,self.where_is_bead.shape[1]), dtype=np.int)
        ), axis=0)
        self.noise_vec = np.concatenate((
            self.noise_vec, np.empty((num_new_beads,2*config.dim))
        ), axis=0)
        self.nbor_list = np.concatenate((
            self.nbor_list, np.zeros(
                (self.nbor_list.shape[0], num_new_beads, self.nbor_list.shape[2]), dtype=np.int)
        ), axis=1)
        self.forces = np.concatenate((
            self.forces, np.zeros((num_new_beads, config.dim))
        ), axis=0)
        return new_chain_number, bi_new_chain

    def mitosis(self, chain_number, sim):
        Lx = config.Lx
        bi_this_chain = self.phonebook[chain_number,1:1+self.phonebook[chain_number,0]]
        old_pos = self.pos[bi_this_chain]
        old_tangents = self.tangent_list[bi_this_chain]
        num_old_beads = bi_this_chain.shape[0]
        num_new_beads = num_old_beads
        new_chain_number, bi_new_chain = self.add_new_chain(
            num_new_beads,
            inherited_label = self.inherited_label[bi_this_chain[-1]]
        )
        new_pos = np.empty((num_old_beads + num_new_beads, config.dim))
        new_tangents = np.empty((num_old_beads + num_new_beads, config.dim))
        seps = np.mod( old_pos[1:] - old_pos[:-1] + Lx/2, Lx) - Lx/2
        dists = np.sum(seps*seps, axis=1)**0.5
        arc_lengths = np.concatenate((np.array([0]),np.cumsum(dists)))
        for i in range(num_old_beads):
            new_arc_len = i * self.reference_rest_length
            idx1 = min( len(old_pos),
                        - 2, np.where(arc_lengths <= new_arc_len)[0][-1] # last old_pos less far along than new bead
                    )
            sep = np.mod(old_pos[idx1 + 1] - old_pos[idx1] + Lx/2,Lx) - Lx/2
            if np.any(sep != 0):
                sep /= (sep.dot(sep)**0.5)
            new_pos[i] = np.mod(old_pos[idx1] + (new_arc_len - arc_lengths[idx1]) * sep, Lx)
            new_tangents[i] = old_tangents[idx1]

        reverse_arc_lengths = arc_lengths[-1] - arc_lengths
        for i in range(num_new_beads):
            new_arc_len = i * self.reference_rest_length
            idx1 = np.where(reverse_arc_lengths <= new_arc_len)[0][0] # first old_pos farther along than new bead
            sep = np.mod(old_pos[idx1 - 1] - old_pos[idx1] + Lx/2,Lx) - Lx/2
            if np.any(sep != 0):
                sep /= (sep.dot(sep)**0.5)
            new_pos[-(i+1)] = np.mod(old_pos[idx1] + (new_arc_len - reverse_arc_lengths[idx1]) * sep, Lx)
            new_tangents[-(i+1)] = -old_tangents[idx1]

        self.pos[bi_this_chain] = new_pos[:num_old_beads]
        self.tangent_list[bi_this_chain] = new_tangents[:num_old_beads]
        self.pos[bi_new_chain] = new_pos[num_old_beads:]
        self.tangent_list[bi_new_chain] = new_tangents[num_old_beads:]

        self.update_grid_list(append_mode = True, start = bi_new_chain[0])
        self.update_nbor_list_with_pop2(self)
        super().reset_force_arrays()

        # set rest lengths back to original size exactly
        self.rest_lengths[chain_number] = self.reference_rest_length * (1 - self.grow_rate_noise * np.random.rand() )
        self.rest_lengths = np.append(
            self.rest_lengths, np.array([self.rest_lengths[chain_number] ])
        )
        return new_chain_number

    def divide_length_factor(self):
        return 2. + self.repel_d_max / ((self.reference_rest_length))

    def grow_and_perhaps_divide(self, chain_number, sim, status):
        grow_length = self.reference_rest_length * self.grow_rate * status.dt_used
        self.rest_lengths[chain_number] += grow_length
        if self.rest_lengths[chain_number] >= 2. * self.reference_rest_length + self.repel_d_max / (self.phonebook[chain_number,0] - 1):
            new_chain_number = self.mitosis(chain_number, sim)

    def grow_and_perhaps_divide_all_chains(self, sim, status):
        num_chains_now = self.num_subpops
        for ci in range(num_chains_now):
            self.grow_and_perhaps_divide(ci, sim, status)

    def set_order_on_chain(self):
        self.order_on_chain = np.empty_like(self.chain_id)
        for chain_contents in self.phonebook:
            this_chain_length = chain_contents[0]
            this_chain_bead_ids = chain_contents[1:]
            for i in range(this_chain_length):
                self.order_on_chain[this_chain_bead_ids[i]] = i
        return self.order_on_chain

    def random_initialization(self, already_know_head_pos = False):
        """ Initialize array of bead positions GIVEN initial positions and orientations
            for the head bead of each chain
        """

        bi = 0
        while bi < self.total_pop:
            if self.order_on_chain[bi] == 0:
                head_index = bi
                if not already_know_head_pos:
                    self.pos[head_index] = np.mod(config.Lx / 2 * (1. + self.pos_init_noise * (2 * np.random.rand(config.dim) - 1)), config.Lx)
                    self.orientations[head_index,0] =  np.mod(np.pi * (np.random.rand() > 0.5) + self.phi_init + 2 * np.pi * self.phi_init_noise * np.random.rand() + np.pi, 2*np.pi) - np.pi
                    if config.dim == 3:
                        self.orientations[head_index,1] = math.acos(2*np.random.rand()-1)
                self.tangent_list[head_index] = unit_vector(self.orientations[head_index])
                ci = self.chain_id[bi]
                bi += 1
            j = 1
            if bi < self.total_pop:
                while self.chain_id[bi] == ci:
                    self.tangent_list[bi] = self.tangent_list[head_index]
                    self.pos[bi] = self.pos[head_index] - self.a_init_factor * self.rest_lengths[ci] * self.tangent_list[head_index] * j
                    bi += 1
                    j += 1
                    if bi > self.total_pop - 1:
                        break
        self.pos = np.mod(self.pos, config.Lx)
        self.initialized = True

    def tangents_to_orientations(self):
        self.orientations[:,0] = np.arctan2(self.tangent_list[:,1],self.tangent_list[:,0])
        if config.dim == 3:
            self.orientations[:,1] = np.arccos(self.tangent_list[:,2])

    def chain_head(self, chain_num):
        return self.phonebook[chain_num,1]

    def chain_members(self, chain_num):
        return self.phonebook[chain_num,1:1+self.phonebook[chain_num,0]]

    def self_interactions(self):
        forces_list = super().self_interactions()
        forces_list.append([
            "self-propulsion",
            self.active_params()
        ])
        # calculate springs first because they update tangents
        if self.longest_chain_length > 1:
            forces_list.insert(0, ["springs", self.spring_params()])

        if self.sliding_partner_maxdist > 0 and self.v0 > 0:
            # remove WCA and self-propulsion
            for force_name in ["WCA", "self-propulsion"]:
                try:
                    idx = [item[0] for item in forces_list].index(force_name)
                    forces_list.pop(idx)
                except:
                    pass
            forces_list.append(
                ["active_sliding_and_WCA", np.append(self.WCA_params, self.active_params())]
            )
        return forces_list

class Equal_Size_Chain_Collection(Chain_Collection):
    """ A Chain_Collection initialized with chains of all equal length """
    def __init__(self, sim, num_chains = -1, chain_length = -1, name = "beads"):
        if num_chains < 0:
            self.num_subpops = sim['n_chains']
        else:
            self.num_subpops = num_chains
        if chain_length < 0:
            self.subpop_size = sim['chain_length']
        else:
            self.subpop_size = chain_length
        self.total_pop = self.num_subpops * self.subpop_size

        self.chain_id = np.asarray(np.floor(np.arange(self.total_pop)/self.subpop_size),dtype=np.int)
        super().__init__(self.num_subpops, self.total_pop, sim, name = name, chain_id = self.chain_id)
        self.b_get_orientations = (self.subpop_size > 1 and self.b_use_tangent_as_propulsion_direction)
        if config.b_announcements:
            print(f'--{self.name}: organizing particles into {self.num_subpops} chains of length {self.subpop_size}')

    # synonym
    def chain_length(self):
        return self.subpop_size


class Motor_Population(Population):
    def __init__(self, sim, num_motors = -1, name = "motors"):
        if num_motors < 0:
            self.total_pop = sim['n_motors']
        else:
            self.total_pop = num_motors
        self.name = name
        super().__init__(self.total_pop,sim,name=name)
        self.D = sim['D_motors']
        self.motor_max_dist = sim['motor_max_dist']
        self.motor_max_dist_sq = self.motor_max_dist**2
        self.b_one_motor_per_bead = sim['b_one_motor_per_bead']
        self.b_motors_dont_diffuse_while_pushing = sim['b_motors_dont_diffuse_while_pushing']
        self.imparts_force = sim['motor_imparts_force']
        self.k_on = sim['k_on']
        self.k_off = sim['k_off']
        if self.D == 0:
            self.fixed = True

    def update_grid_list(self, append_mode = False, overwrite = False):
        while not updates.update_grid_list(
                self.pos, self.grid_list, self.where_is_bead,
                config.Lx, append_mode = append_mode, overwrite = overwrite
        ):
            self.grid_list = np.concatenate((np.zeros_like(self.grid_list),)*2, axis=-1)
            overwrite = True

    def self_interactions(self):
        if self.D > 0:
            return [
                [
                    "diffuse",
                    np.array([
                        self.b_motors_dont_diffuse_while_pushing],
                        dtype=np.float
                    )
                ]
            ]
        else:
            return [ ]


    def size_scale(self):
        return self.motor_max_dist



class Shells(Equal_Size_Chain_Collection):
    def __init__(self, sim, num_chains = -1, chain_length = 1, name = "shells", qq_strength = 1., qq_cutoff = 2.):
        super().__init__(sim, num_chains = num_chains, chain_length = chain_length, name = name)
        self.qq_strength = qq_strength
        self.qq_cutoff = qq_cutoff

    def self_interactions(self):
        forces_list = super().self_interactions()
        forces_list.append(
            [
                "quadrupole_quadrupole", np.array([self.qq_strength,self.qq_cutoff])
            ]
        )
        return forces_list

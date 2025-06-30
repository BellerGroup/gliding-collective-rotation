#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os, glob, sys, importlib
from time import localtime, strftime
from socket import gethostname
from time import time as clocktime
import populations, cline, config, control
import output_methods as io
from numba import njit,jit

"""
(Under construction) Suite of analysis tools including the ability to create
new plots from stored data, as "snapshots" or "animations". Most easily used
in the Python interpreter.
"""

default_binsize = 1
plt.ion()


def recall(filelabel, filenumber = -1, resultspath = './Results/', imgpath = './Images/', pop_names = ["beads"]):
    """ Load a saved population from a saved timepoint from a given run, along with
        the simulation parameters as a "sim" dictionary.
    """
    fileprefix = resultspath + filelabel
    sim = cline.read_cl_record(fileprefix)
    config.dim = sim["dim"]
    if filenumber < 0:
        # default to most recent
        filenumber = int(sorted(glob.glob(fileprefix + '*.dat'), key=os.path.getmtime)[-1].split('.dat')[0].split('_')[-1])
        print(f'Loading last saved timestep (filenumber {filenumber})')
    stepstring = io.zero_padded_string(filenumber,sim['n_digits_stepnum_str'])
    pops = {}
    for pop_name in pop_names:
        filename = fileprefix + '_' + pop_name + '_' + stepstring + '.dat'

        if os.path.exists(filename):
            with open(filename,'r') as file_in:
                lines = file_in.readlines()
            headers = lines[1].split('#')[1].split()
            data = lines[2:]
            if 'chain_id' in headers:
                chain_id = [ int(float(line.split(' ')[headers.index('chain_id')])) for line in data ]
                num_chains = max(chain_id) + 1
                chain_length = sum( [ 1 for i in range(len(chain_id)) if chain_id[i] == min(chain_id) ] )
                where_active = np.array([ int(float(line.split(' ')[5])) for line in data ])
            if 'inherited_label' in headers:
                inherited_label = [ int(float(line.split(' ')[headers.index('inherited_label')])) for line in data ]

            pos = []
            for header_item in ['x','y','z']:
                if header_item in headers:
                    pos.append([ float(line.split(' ')[headers.index(header_item)]) for line in data ])
            orientations = []
            for header_item in ['phi','theta']:
                if header_item in headers:
                    orientations.append([ float(line.split(' ')[headers.index(header_item)]) for line in data ])


            first_line_vals = lines[0].split()[1:]
            class_string = first_line_vals[first_line_vals.index('class') + 1]
            stepnum = int(first_line_vals[first_line_vals.index('stepnum') + 1])
            filenumber = int(first_line_vals[first_line_vals.index('filenumber') + 1])
            t = float(first_line_vals[first_line_vals.index('t') + 1])

            total_pop = len(pos[0])
            class_kwargs = {"name": pop_name}
            class_name = class_string.split('populations.')[-1]
            if class_name == 'Equal_Size_Chain_Collection':
                class_args = (sim,)
                class_kwargs["num_chains"] = num_chains
                class_kwargs["chain_length"] = chain_length
            elif class_name == 'Chain_Collection':
                class_args = (num_chains, total_pop, sim)
            elif class_name in ['Population', 'WCA_Population', 'Oriented_WCA_Population']:
                class_args = (total_pop, sim)
            elif class_name == 'Motor_Population':
                class_args = (sim,)
                class_kwargs["num_motors"] = total_pop
            else:
                print("Unrecognized class: " + class_string)
                exit()
            config.b_announcements = False
            pop = eval(class_string + '(*' + str(class_args) + ', **' + str(class_kwargs) + ')')
            pop.pos = np.array(pos).T
            pop.orientations = np.array(orientations).T
            pop.initialized = True # remember positions, etc.
            pop.where_active = where_active
            if 'chain_id' in headers:
                if pop.orientations.shape[1] == 1:
                    pop.tangent_list = np.column_stack((np.cos(pop.orientations[:,0]), np.sin(pop.orientations[:,0])))
                elif pop.orientations.shape[1] == 2:
                    sin_theta = np.sin(pop.orientatations[:,1])
                    pop.tangent_list = np.column_stack(( sin_theta * np.cos(pop.orientations[:,0]),
                                                     sin_theta * np.sin(pop.orientations[:,0]),
                                                     np.cos(pop.orientations[:,1])
                                                   ))
                pop.chain_id = np.array(chain_id)
                pop.inherited_label = np.array(inherited_label)

            pops[pop_name] = pop
        else:
            return {"sim":False, "pops":False, "stepnum":False, "t":False, "filenumber":False}

    return {"sim":sim, "pops":pops, "stepnum":stepnum, "t":t, "filenumber":filenumber}

def snapshot(filelabel, filenum = -1, replacements = {}, resultspath = './Results/',
    save = False, imgpath = './Images/', img_file_digits = 6, pop_plots = None,
    figure = None, pop_names = ["beads"], pause_time = 1e-8):
    """
    Create a snapshot figure from saved data.
    """
    plt.ion()
    sim, pops, stepnum, t, filenumber = recall(filelabel, filenum, resultspath = resultspath, imgpath = imgpath, pop_names = pop_names).values()
    if sim != False:
        sim.update(replacements)
        if pop_plots == None or figure == None:
            pop_names = list(pops.keys())
            figure, beads_plots, motors_plots = io.create_plot_utility(sim, [ pops[pop_name] for pop_name in pop_names ] )
            pop_plots = {}
            for pni in range(len(pops.keys())):
                pop_plots[pop_names[pni]] = (beads_plots + motors_plots)[pni]
        else:
            for pop_name in pops.keys():
                io.update_beads_plot(sim,pops[pop_name],pop_plots[pop_name])

        io.redraw(figure, sim)
        plt.pause(pause_time)
        if save:
            if config.dim == 3:
                for pop_plot in pop_plots.values():
                    pop_plot.set_alpha(io.alpha_3D) # colors don't render in saved 3D figure
                    #if some property like this is not adjusted here (matplotlib bug)
                    # https://github.com/matplotlib/matplotlib/issues/9725
            imgfilename = imgpath + filelabel + '_' + io.zero_padded_string(filenumber,img_file_digits) + '.png'
            plt.savefig(imgfilename, metadata = io.set_plot_metadata(filelabel, stepnum, t), facecolor=figure.get_facecolor())
        return pop_plots, figure
    else:
        return [None]*2

def animate(filelabel, start= 0, stop=-1, replacements = {}, pause_time = 0.4,
    wait = False, resultspath = './Results/', save_imgs = False, save_vid = False,
    imgpath = './Images/', pop_names = ['beads'], frame_rate = 5):
    """
    Create an animation from a series of snapshots.
    """
    fileprefix = resultspath + filelabel
    filenum = start
    pop_plots = None
    figure = None
    while filenum < stop or stop < 0:

        pop_plots, figure = snapshot(filelabel, filenum, replacements = replacements,
                        resultspath = resultspath, save = save_imgs or save_vid, imgpath = imgpath, pop_plots = pop_plots, figure = figure,
                        pop_names = pop_names, pause_time = pause_time)

        if figure == None:
            print(f'Reached file number {filenum} with no existing state file under {fileprefix} for at least one of the populations {pop_names}')
            break
        if wait:
            plt.waitforbuttonpress()
        filenum += 1

    if save_vid:
        movie_from_snapshot(filelabel, imgpath = imgpath, frame_rate = frame_rate)


def movie_from_snapshot(filelabel, imgpath = 'Images/', frame_rate = 5):
    print(
    f"ffmpeg -framerate {frame_rate} -pattern_type glob -i \'{imgpath + filelabel}_*.png\' -c:v libx264 -pix_fmt yuv420p {imgpath + filelabel}_anim.mp4"
        )
    os.system(
        f"ffmpeg -framerate {frame_rate} -pattern_type glob -i \'{imgpath + filelabel}_*.png\' -c:v libx264 -pix_fmt yuv420p {imgpath + filelabel}_anim.mp4"
    )

def resume(filelabel, filenum = -1, resultspath = 'Results/', pop_names = ['beads']):
    sim, pops, stepnum, t, filenumber= recall(filelabel, filenum, resultspath = resultspath, pop_names = pop_names).values()
    for pop in pops.values():
        pop.initialized = True
    control.run_sim(sim, list(pops.values()), start_t=t, start_stepnum = stepnum, start_filenumber = filenumber)

@jit(nopython=True, fastmath = True)
def true_sep(pt1, pt2, system_size):
    """ Correction for periodic boundary conditions in separation vector """
    return np.mod( pt2 - pt1 + system_size/2, system_size ) - system_size/2

@jit(nopython=True, fastmath = True)
def true_dist(pt1, pt2, system_size):
    """ Correction for periodic boundary conditions in separation distance """
    sep = true_sep(pt1,pt2,system_size)
    return np.sqrt( np.sum(sep**2) )

@jit(nopython=True, fastmath = True)
def unit_vectors_dot(angle1, angle2):
    return np.cos(angle1)*np.cos(angle2) + np.sin(angle1)*np.sin(angle2)

@jit(nopython=True, fastmath = True)
def directors_dot(angle1, angle2):
    return np.abs(unit_vectors_dot(angle1, angle2))

# General template for instantaneous correlation functions.
# "func" must be a function lambda beads, ci, cj
# where ci and cj are two chain numbers
def correlation_func(beads_record, func, binsize=0.25, norm = 'bin', pop_name = "beads"):
    Lx = beads_record["sim"]["Lx"]
    beads = beads_record["pops"][pop_name]
    bin_ranges = np.arange(0,int(Lx/2)*np.sqrt(2),binsize)
    bin_sums = np.zeros(bin_ranges.shape[0])
    bin_counts = np.zeros_like(bin_sums,dtype=np.int)

    if pop_name == 'beads':
        for ci in range(beads.num_chains()):
            ci_COM = chain_COM(beads,ci,Lx)
            for cj in range(ci):
                dist = true_dist(ci_COM, chain_COM(beads,cj,Lx),Lx)
                bin_num = int(dist/binsize)
                bin_sums[bin_num] += 2 * func(beads, ci, cj)
                bin_counts[bin_num] += 2

    if pop_name == 'motors':
        for ci in range(beads.total_pop): #motor.total_pop. here "beads" means motor from the asignment above
            for cj in range(ci):
                dist = true_dist(beads.pos[ci],beads.pos[cj], Lx)
                bin_num = int(dist/binsize)
                bin_sums[bin_num] += 2 * func(beads, ci, cj)
                bin_counts[bin_num] += 2

    if norm == 'bin':
        # for two-point correlations, averaging over values in each bin
        denominator = np.maximum(bin_counts,1)
    elif norm == 'all':
        # 2 \pi r dr \rho
        number_density = np.sum(bin_counts) / (Lx*Lx)
        denominator = number_density * 2 * np.pi * binsize * ( bin_ranges + binsize / 2)
    else:
        print(f"Invalid normalization norm = {norm}")
    bin_values = bin_sums / denominator
    answer = np.stack((bin_ranges, bin_values),axis=1)
    return answer[np.where(answer[:,0] < Lx/2)]

def g_of_r(beads_record, binsize=default_binsize, pop_name = "beads"):
    """ radial pair correlation function """
    return correlation_func(beads_record,
        lambda beads, ci, cj: 1,
        norm = 'all',
        binsize=binsize, pop_name = pop_name
    )

def Cvv(beads_record, binsize=default_binsize, pop_name = "beads"):
    """ Instantaneous velocity-velocity correlation function """
    return correlation_func(beads_record,
        lambda beads, ci, cj:
            unit_vectors_dot(
                io.calc_avg_phi_chain(beads,ci),
                io.calc_avg_phi_chain(beads,cj)
            ),
        norm = 'bin',
        binsize=binsize, pop_name = pop_name
    )

def Cnn(beads_record, binsize=default_binsize):
    """ Instantaneous director-director correlation function """
    return correlation_func(beads_record,
        lambda beads, ci, cj:
            directors_dot(
                io.calc_avg_phi_chain(beads, ci, n_fold_symmetry = 2),
                io.calc_avg_phi_chain(beads, cj, n_fold_symmetry = 2)
            ),
            norm = 'bin',
            binsize=binsize
    )

def lineplot(arr):
    plt.ion()
    plt.plot(arr[:,0],arr[:,1])

def scatterplot(arr):
    plt.ion()
    plt.scatter(arr[:,0],arr[:,1])

def reload(module='analysis'):
    """ Reloads a module (not a file) """
    return importlib.reload(sys.modules[module])


def chain_COM(beads,chain_num,system_size):
    """ Center of mass of a chain """
    chain_bi = beads.phonebook[chain_num,1:1+beads.phonebook[chain_num,0]]
    head_pos = beads.pos[chain_bi[0]]
    return np.mod(head_pos + np.sum(true_sep(beads.pos[chain_bi],head_pos,system_size),axis=0),system_size)

def nematic_orderparameter(beads_record, pop_name = "beads"):
    """Global nematic order parameter for a timestep """
    beads = beads_record["pops"][pop_name]
    n_chains = beads.num_chains()
    ex = 0.0; ey = 0.0; Q11=0.0; Q12=0.0;
    Q11_full = 0.0; Q12_full = 0.0;
    order_parameter = 0.0; director_angle = 0.0;
    evec = 0.0;
    for ci in range(n_chains):
        avg_phi = io.calc_avg_phi_chain(beads,ci)
        ex = np.cos(avg_phi)
        ey = np.sin(avg_phi)
        Q11 = 2*ex*ex - 1.0
        Q12 = 2*ex*ey
        Q11_full += Q11
        Q12_full += Q12
    order_parameter = (np.sqrt((Q11_full**2) + (Q12_full**2)))
    evec = (order_parameter - Q11_full)/Q12_full
    director_angle = np.arctan(evec)
    return order_parameter/n_chains, director_angle

def polar_orderparameter(beads_record, pop_name = "beads"):
    """Global nematic order parameter for a timestep """
    beads = beads_record["pops"][pop_name]
    n_chains = beads.num_chains()
    ex = 0.0; ey = 0.0; ex_sum = 0.0; ey_sum = 0.0;
    order_parameter = 0.0;
    for ci in range(n_chains):
        avg_phi = io.calc_avg_phi_chain(beads,ci)
        ex = np.cos(avg_phi)
        ey = np.sin(avg_phi)
        ex_sum += ex
        ey_sum += ey
    order_parameter = (np.sqrt((ex_sum**2) + (ey_sum**2)))/n_chains
    return order_parameter

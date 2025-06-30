#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import numpy as np
import os
from time import localtime, strftime
from socket import gethostname
from time import time as clocktime
from sys import stdout
import populations, cline, config

"""
Displaying plots, saving plots, saving data, and printing status to standard out "ticker"
"""

alpha_3D = 0.6
alpha_2D = None
edgecolor = [0,0,0,0.1]
cb_frac = 0.1 # size of colorbar
default_bead_color = 'cyan'

def create_plot(sim,pops):

    if not sim['b_display_plots']:
        from matplotlib import use as mpluse
        mpluse('Agg')
        sim['plot_every'] *= max(sim['save_every_nth_plot'],1)
        sim['save_every_nth_plot'] = 1
    if sim['plot_every'] > 0:
        print("--Setting up the plot")

        return create_plot_utility(sim, pops)

def create_plot_utility(sim, pops):
    WCA_pops = [ pop for pop in pops if isinstance(pop,populations.WCA_Population)]
    motors_pops = [ pop for pop in pops if isinstance(pop,populations.Motor_Population)]
    plot_range_padding = min([pop.repel_d_max for pop in WCA_pops])
    figure, ax = create_figure(sim, plot_range_padding)
    beads_plots = [create_beads_plot(sim,pop,ax) for pop in WCA_pops]
    motors_plots = [create_motors_plot(sim,pop) for pop in motors_pops]

    plt.gcf().set_facecolor('black')
    txt_clr = 'white'
    mpl.rcParams['text.color'] = txt_clr
    mpl.rcParams['axes.labelcolor'] = txt_clr
    mpl.rcParams['xtick.color'] = txt_clr
    mpl.rcParams['ytick.color'] = txt_clr
    ax.set_edgecolor = 'white'
    # Set the spines to be white.
    for spine in ax.spines:
        ax.spines[spine].set_color(txt_clr)

    # Set the ticks to be white
    for axis in ('x', 'y'):
        ax.tick_params(axis=axis, color=txt_clr)

    # Set the tick labels to be white
    for tl in ax.get_yticklabels():
        tl.set_color(txt_clr)
    for tl in ax.get_xticklabels():
        tl.set_color(txt_clr)

    if sim['b_show_colorbar']:
        if sim['color_by'] in [
            'chain_orientation','bead_orientation','chain_director'
        ]:
            mpl.rcParams['text.usetex'] = True
            axins = (lambda ht:ax.inset_axes([1.01,1-ht,ht,ht]))(0.1)
            angle_multiplier = 2 if sim['color_by']=='chain_director' else 1
            polar_colorbar(axins, angle_multiplier)
            plt.tight_layout(w_pad=20)
            mpl.rcParams['text.usetex'] = False
        elif sim['color_by'] in ['none','False']:
            pass
        else:
            if sim['color_by'] == 'population':
                color_bars = (
                    [figure.colorbar(
                        beads_plots[0],
                        shrink = 0.2 * len(config.pop_name_list),
                        fraction = cb_frac, drawedges = False,
                        pad = 0.02,  **kwargs_to_colorbar(sim, WCA_pops[0])
                        )
                    ]
                )
                color_bars[0].set_ticks(np.arange(0,len(config.pop_name_list)))
                color_bars[0].set_ticklabels(config.pop_name_list)
                plt.setp(color_bars[0].ax.get_yticklabels(),rotation=90, verticalalignment='center',size=11)
            else:
                color_bars = (
                    [figure.colorbar(
                        beads_plots[bpi], shrink = min(0.5, 0.1 * len(kwa["boundaries"])), fraction = cb_frac, drawedges = False,
                        pad = 0.02, **kwa
                        ) for bpi, kwa in enumerate([kwargs_to_colorbar(sim, WCA_pops[bpi2]) for bpi2 in range(len(WCA_pops))])
                    ]
                )

            for cb, pop in zip(color_bars, WCA_pops):
                if len(color_bars) > 1:
                    label = pop.name + ': '
                else:
                    label = ''
                label += sim['color_by']
                cb.set_label(label=label,size=12,weight='bold')
                cb.solids.set_edgecolor("face")

    plt.ion()
    return figure, beads_plots, motors_plots


def polar_colorbar(axpc, angle_multiplier):
    u = np.arange(-np.pi, np.pi, 2*np.pi/50)
    v = 0.3
    x = v*np.cos(u)
    y = v*np.sin(u)
    axpc.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, left=False, bottom=False)
    axpc.set_aspect('equal')
    axpc.scatter(
        x,y,s=50, c=(
            -np.pi+np.mod(np.pi+angle_multiplier*u,2*np.pi)
        ) ,cmap='hsv'
    )
    # textsep = 1.1
    # for i in range(4):
    #     axpc.text(
    #             textsep*[1,0,-1,0][i],
    #             textsep*[0,1,0,-1][i],
    #             [r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$'][i],
    #             size=13, horizontalalignment='center', verticalalignment='center', color='white'
    #     )
    pad = 0.2
    bound = v + pad
    axpc.set_xlim(-bound,bound)
    axpc.set_ylim(-bound,bound)
    axpc.set_facecolor('black')

def kwargs_to_colorbar(sim, pop):
    n_ticks = 8
    n_colors_gradient = 1000 # large number avoids ugly line artifacts that
        # can occur from automatic colorbar creation
    cby = sim['color_by']
    cvals = tks = fmt = None
    if cby in [
        'chain_id', 'order_on_chain', 'where_active', 'inherited_label',
        'x', 'y', 'z', 'population'
    ]:

    ## color by chain
        if cby == 'chain_id':
            cvals = np.arange(-0.5,1+np.max(pop.chain_id))
        elif cby == 'order_on_chain':
            cvals = np.arange(-0.5,1+np.max(pop.order_on_chain))
        elif cby == 'where_active':
            cvals = np.arange(-0.5, 2)
        elif cby == 'inherited_label':
            cvals = np.arange(0.5, 2+np.max(pop.inherited_label))
        elif cby == 'population':
            cvals = (
                np.arange(-0.5,len(config.pop_name_list))
            )
        else:
            cvals = np.arange(0,
                config.Lx*(1+1/n_colors_gradient), config.Lx/n_colors_gradient)
        tks = [ int(item) for item in
            np.arange(
                np.min(cvals),
                np.max(cvals) + 1, (np.max(cvals) - np.min(cvals))/(n_ticks - 1)
            )
        ]
        fmt = '%d'
    else:
        fmt = '%.1f'

        if cby in ['chain_orientation', 'bead_orientation']:
            cvals =  np.arange(-np.pi, np.pi, 2*np.pi/n_colors_gradient)
            tks = np.arange(-3,4,1)
        elif cby == 'chain_director':
            cvals = np.arange(-np.pi/2, np.pi/2, np.pi/n_colors_gradient)
            tks = np.arange(-1.5, 1.6, 0.5)
        elif cby == 'polar_angle':
            cvals = np.arange(0,np.pi,np.pi/n_colors_gradient)
            tks = np.arange(0, np.pi, 0.5)
    return {"boundaries":cvals, "ticks":tks, "format":fmt}


def create_figure(sim,plot_range_padding):
    if config.dim == 2:
        figure = plt.figure(figsize=(sim['figsize'], 0.95 * (1-cb_frac) * sim['figsize']))
        ax = plt.axes()
        ax.set_xlim(-plot_range_padding, config.Lx+plot_range_padding)
        ax.set_ylim(-plot_range_padding, config.Lx+plot_range_padding)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        plt.tight_layout()
    elif config.dim == 3:
        figure = plt.figure(figsize=(sim['figsize'],sim['figsize']))
        ax = figure.add_subplot(111, projection='3d', proj_type='ortho')
        ax.set_xlim(-plot_range_padding, config.Lx+plot_range_padding)
        ax.set_ylim(-plot_range_padding, config.Lx+plot_range_padding)
        ax.set_zlim(-plot_range_padding, config.Lx+plot_range_padding)

        ax.set_facecolor('black')

        ax.w_xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0 ))
        ax.w_yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        ax.w_zaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
        c = (0.2, 0.2, 0.2, 1.0)
        ax.xaxis.pane.set_edgecolor(c)
        ax.yaxis.pane.set_edgecolor(c)
        ax.zaxis.pane.set_edgecolor(c)
        figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.tick_params(colors=(0.5,0.5,0.5,1.0))
    return figure, ax


def create_beads_plot(sim,beads,ax):
    sphere_size_factor = 72 if config.dim == 2 else 100
    args = (beads.pos[:,0], beads.pos[:,1])
    kwargs = {
        'c': np.zeros((beads.how_many())),
        'cmap': 'hsv',
        'vmin': -np.pi,
        'vmax': np.pi,
        's': (
            (beads.repel_d_max+beads.WCA_shift)
            * sim['figsize']*sphere_size_factor/config.Lx
        )**2 / (0.5*np.pi)
    }

    if config.dim == 3:
        args += (beads.pos[:,2],)
        kwargs["marker"] = "."
        kwargs["edgecolor"] = "none"
        alpha = alpha_3D
    else:
        alpha = alpha_2D
    beads_plot = ax.scatter(*args, **kwargs)

    beads_plot.set_alpha(alpha)
    set_beads_colors(sim,beads,beads_plot)
    beads_plot.set_edgecolor(edgecolor)

    return beads_plot


def create_motors_plot(sim,motors):
    return plt.scatter(motors.pos[:,0],motors.pos[:,1],
                                 s=.4, c='white')


def set_beads_colors(sim,beads,beads_plot):
    """ Update the colors of each bead, each time the plot is redrawn, according
        to the rule chosen in the command line
    """
    ### coloring options ###
    if sim['color_by'] == 'chain_orientation':
    ## color by average orientation of chain
        beads_plot.set_clim(-np.pi,np.pi)
        beads_plot.set_array(calc_avg_phi(beads))
    elif sim['color_by'] == 'chain_director':
        beads_plot.set_clim(-np.pi/2.,np.pi/2.)
        beads_plot.set_array(calc_avg_phi(beads, n_fold_symmetry = 2))
    elif sim['color_by'] == 'bead_orientation':
    ## color by bead's orientation
        beads_plot.set_clim(-np.pi,np.pi)
        beads_plot.set_array( np.arctan2( beads.tangent_list[:,1], beads.tangent_list[:,0] ) )
    elif sim['color_by'] == 'polar_angle' and config.dim == 3:
        beads_plot.set_clim(-np.pi/2, np.pi)
        # Note: the interval [-pi/2,0) is not in the range of np.arccos()
        # This is intentionally included in clim so that a cyclic color space
        # can be used without appearing cyclic for the polar angle.
        # For non-cyclic color maps, it's safe to set the lower clim to 0.
        beads_plot.set_array(np.arccos( beads.tangent_list[:,2] ) )
    elif sim['color_by'] == 'chain_id':
    ## color by chain
        beads_plot.set_clim(0,beads.num_chains())
        beads_plot.set_array(beads.chain_id)
    elif sim['color_by'] == 'population':
        pop_index = config.pop_name_list.index(beads.name)
        beads_plot.set_clim(0, len(config.pop_name_list))
        beads_plot.set_array(pop_index * np.ones(beads.total_pop))
    elif sim['color_by'] == 'order_on_chain':
    ## color by bead number on chain
        beads_plot.set_clim(0,beads.chain_length())
        beads_plot.set_array(beads.order_on_chain)
    elif sim['color_by'] == 'where_active':
    ## color by which beads are actively self-propelling
        beads_plot.set_clim(-1,1)
        beads_plot.set_array( beads.where_active > 0 )
    elif sim['color_by'] == 'inherited_label':
        beads_plot.set_clim(0,np.max(beads.inherited_label) + 1)
        beads_plot.set_array( beads.inherited_label )
    elif sim['color_by'] == 'x':
        beads_plot.set_clim(0, config.Lx)
        beads_plot.set_array( beads.pos[:,0] )
    elif sim['color_by'] == 'y':
        beads_plot.set_clim(0, config.Lx)
        beads_plot.set_array( beads.pos[:,1] )
    elif sim['color_by'] == 'z' and config.dim == 3:
        beads_plot.set_clim(0, config.Lx)
        beads_plot.set_array( beads.pos[:,2] )
    elif sim['color_by'] in ['none', 'False']:
        beads_plot.set_color(default_bead_color)
        beads_plot.set_edgecolor(edgecolor)
    else:
        print('Invalid color_by = ', sim['color_by'])
        exit()


def update_beads_plot(sim,beads,beads_plot):
    if config.dim == 2:
        beads_plot.set_offsets(beads.pos)
    elif config.dim == 3:
        beads_plot._offsets3d = juggle_axes(beads.pos[:,0], beads.pos[:,1], beads.pos[:,2], 'z')
    set_beads_colors(sim,beads,beads_plot)
    # if max(beads.where_active) > min(beads.where_active):
        # beads_plot.set_edgecolors([
        #     'gray' if is_active==-1 else 'black'
        #     for is_active in beads.where_active
        # ])
        # beads_plot.set_alpha([
        #     0.5 if is_active==-1 else 1
        #     for is_active in beads.where_active
        # ])

def redraw(figure,sim):
    """ refresh plot """
    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.pause(sim['plot_pause_time'])


def update_motors_plot(sim,motors,motor_plot):
    motor_plot.set_offsets(motors.pos)


def zero_padded_string(the_number,n_digits):
    """ Add zeros to the front of the file number for saving data or plots, so
        all filenumbers have the same number of numerical digits.
    """

    return ('0'*n_digits)[:-len(str(the_number))] \
                + str(the_number)


def update_plot(sim, pops, status, figure, beads_plots, motors_plots):
        # update the plot data

        WCA_pops = [ pop for pop in pops if isinstance(pop,populations.WCA_Population)]
        motors_pops = [ pop for pop in pops if isinstance(pop,populations.Motor_Population)]

        for pop_i in range(len(WCA_pops)):
            update_beads_plot(sim,WCA_pops[pop_i],beads_plots[pop_i])

        for pop_i in range(len(motors_pops)):
            update_motors_plot(sim,motors_pops[pop_i],motors_plots[pop_i])

        redraw(figure,sim)

        # save plot as png
        if sim['save_every_nth_plot'] > 0 and \
                round(status.stepnum/sim['plot_every']) % sim['save_every_nth_plot'] == 0:
            imgfilename =  sim['imgfileprefix']
            # prepend zeros to stepnum string

            if not sim['b_save_most_recent_img_only']:
                stepnum_str = zero_padded_string(status.stepnum,sim['n_digits_stepnum_str'])
                imgfilename += '_' + stepnum_str
            imgfilename += '.png'

            plt.savefig(imgfilename,
                        metadata = set_plot_metadata(sim["filelabel"], status.stepnum, status.t)
            )


def set_plot_metadata(filelabel,stepnum,t):
    return ({
    'Title': filelabel  + f'step {stepnum}',
    'Description': f't {t:.5f}',
    'Creation Time': strftime("%c",localtime()),
    'Software': os.path.basename(__file__),
    'Source': gethostname()
    })


def ticker(sim, status):
    """ stdout status update """
    dt_avg = (status.t - status.t_prev) / sim['ticker_every']
    clocktime_now = clocktime()
    clocktime_per_step = (clocktime_now - status.clocktime_prev) / sim['ticker_every']
    if status.t > status.t_prev:
        clocktime_per_simtime = (clocktime_now - status.clocktime_prev) / (status.t-status.t_prev)
    else:
        clocktime_per_simtime = 0.
    print(f'\t{status.stepnum}\t\t{status.t:.2f}\t\t{dt_avg:.4e}\t{clocktime_per_step:.4e}\t{clocktime_per_simtime:.4e}\t{status.max_forces_strength:.4e}')
    stdout.flush()
    status.t_prev = status.t
    status.clocktime_prev = clocktime_now
    status.max_force_strength_sum = 0
    return


def first_ticker(sim,status):
    """ top lines of stdout status update """
    print('\n\tstep\t\tt\t\tdt\t\twalltime/step\twalltime/dt\tmax force')
    print('\t' + ('-----\t\t')*6)
    print(f'\t{status.stepnum}\t\t{status.t:.2f}\t\t{sim["dt0"]:.3e}\t --- \t\t --- \t\t ---')
    stdout.flush()
    status.t_prev = status.t
    status.clocktime_prev = clocktime()
    return


def save_state(sim,pops,status):
    for pop in pops:
        outfilename = sim['outfileprefix'] + '_' + pop.name
        # prepend zeros to stepnum string
        if not sim['b_save_most_recent_only']:
            filenumber_str = ('0'*sim['n_digits_stepnum_str'])[:-len(str(status.filenumber))] \
                            + str(status.filenumber)
            outfilename += '_' + filenumber_str
        outfilename += '.dat'
        head_string = f'filenumber {status.filenumber} stepnum {status.stepnum} t {status.t:.5f} ' + \
                     ''.join(''.join(str(type(pop)).split("<")[-1].split(">")).split("\'")) + '\n'
        if isinstance(pop, populations.Chain_Collection):
            pop.tangents_to_orientations()
            if config.dim == 2:
                np.savetxt(outfilename,
                    np.column_stack((pop.chain_id, pop.inherited_label, pop.pos[:,0], pop.pos[:,1],
                         pop.orientations[:,0], pop.where_active)),
                        fmt = ['%d', '%d', '%.8e', '%.8e', '%.8e', '%d'],
                        header = head_string + \
                          f'chain_id inherited_label x y phi where_active inherited_label'
                    )
            elif config.dim == 3:
                np.savetxt(outfilename,
                    np.column_stack((pop.chain_id, pop.inherited_label, pop.pos[:,0], pop.pos[:,1], pop.pos[:,2],
                         pop.orientations[:,0], pop.orientations[:,1], pop.where_active)),
                        fmt = ['%d', '%d', '%.8e', '%.8e', '%.8e', '%.8e', '%.8e', '%d'],
                        header = head_string + \
                          f'chain_id inherited_label x y z phi theta where_active inherited_label'
                    )

        elif isinstance(pop, populations.Motor_Population):
            if config.dim == 2:
                np.savetxt(outfilename,
                            np.column_stack((pop.pos[:,0], pop.pos[:,1], pop.where_active
                                             )),
                            fmt = ['%.8e', '%.8e', '%d'],
                            header = head_string + \
                                      f'x y did_motor_push'
                          )
            elif config.dim == 3:
                np.savetxt(outfilename,
                            np.column_stack((pop.pos[:,0], pop.pos[:,1], pop.pos[:,2], pop.did_motor_push.astype(np.int)
                                             )),
                            fmt = ['%.8e', '%.8e', '%.8e', '%d'],
                            header = f'filenumber {status.filenumber} stepnum {status.stepnum} t {status.t:.5f}\n' + \
                                      f'x y z did_motor_push'
                          )
        else:
            if config.dim == 2:
                np.savetxt(outfilename,
                            np.column_stack((pop.pos[:,0], pop.pos[:,1]
                                             )),
                            fmt = ['%.8e', '%.8e'],
                            header = f'filenumber {status.filenumber} stepnum {status.stepnum} t {status.t:.5f}\n' + \
                                      f'x y'
                          )
            elif config.dim == 3:
                np.savetxt(outfilename,
                            np.column_stack((pop.pos[:,0], pop.pos[:,1], pop.pos[:,2],
                                             )),
                            fmt = ['%.8e', '%.8e', '%.8e'],
                            header = head_string + \
                                      f'x y z'
                          )
    status.filenumber += 1
    return


def chain_bead_indices(beads, chain_num):
    return beads.phonebook[chain_num,1:1+beads.phonebook[chain_num,0]]


def calc_avg_phi_chain(beads, chain_num, n_fold_symmetry = 1):
    """ Calculate a chain's average orientation (azimuthal angle) """
    bi_this_chain = chain_bead_indices(beads, chain_num)
    if n_fold_symmetry == 1:
        avg_phi = np.arctan2(
            np.sum(beads.tangent_list[bi_this_chain,1]),
            np.sum(beads.tangent_list[bi_this_chain,0])
        )
    else:
        phi_list = np.arctan2(
            beads.tangent_list[bi_this_chain,1],
            beads.tangent_list[bi_this_chain,0]
        )
        avg_phi = np.arctan2(
            np.sum(np.sin(n_fold_symmetry*phi_list)),
            np.sum(np.cos(n_fold_symmetry*phi_list))
        ) / n_fold_symmetry

    return avg_phi


def calc_avg_phi(beads, n_fold_symmetry = 1):
    """ For color in plot, calculate average orientation angle of each chain"""
    avg_phi_beads = np.zeros((beads.how_many()))
    for ci in range(beads.num_chains()):
        avg_phi_chain = calc_avg_phi_chain(beads, ci, n_fold_symmetry = n_fold_symmetry)
        # associate same average value to all beads on chain:
        avg_phi_beads[chain_bead_indices(beads, ci)] = avg_phi_chain
    return avg_phi_beads

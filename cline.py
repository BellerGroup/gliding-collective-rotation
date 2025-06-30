#!/usr/bin/env python3
import argparse
import config
import numpy as np
import defaults

"""
Process command line inputs, calculate some derived quantities from them, and print
warnings for certain bad combinations of parameters.
"""

def cline():
    """ Process command line inputs and other optional parameters with their
        default values
    """
    sim, help = defaults.set_defaults()

    arg_names = [ item[0] for item in sim.items() ]

    arg_shortnames = []

    for i in range(len(arg_names)):
    	checked_not_used_before = False
    	ii = 0
    	while not checked_not_used_before and ii < 5:
            True
            # argnames that should be same as shortnames:
            if arg_names[i] in ['D', 'Dr', 'T']:
                arg_shortname = arg_names[i]
            else:
                arg_shortname = ''.join([ item[0:1+min(ii,len(item))] for item in arg_names[i].split('_') \
        							if item != 'b'] ).lower()
            used_before = False
            for j in range(i):
            	if arg_shortname == arg_shortnames[j]:
            		used_before = True
            if not used_before:
            	checked_not_used_before = True
            	arg_shortnames.append(arg_shortname)
            else:
            	ii += 1
            	if ii == 5: # 5 is arbitrary
            		print('Error: cannot find a suitable shortname for variable' + arg_names[i])
            		exit()
            	# else stay in while loop and try again

    parser = argparse.ArgumentParser()
    for i in range(len(arg_names)):
    	arg_name = arg_names[i]
    	arg_shortname = arg_shortnames[i]
    	arg_default_val = sim[arg_name]
    	arg_type = type(arg_default_val)
    	if arg_name in help.keys():
            help_string = help[arg_name]
    	else:
            help_string = ''
    	if arg_type == bool:
            arg_type = ( lambda x: True if x == 'True' else (False if x == 'False' else (int(x) != 0) ) )
    	parser.add_argument("-" + arg_shortname, "--" + arg_name, type=arg_type, \
    						default=arg_default_val, help = help_string)

    args = parser.parse_args()

    sim, cline_string = args_to_dict(arg_names, args)

    print('')
    for line in cline_string.split('--')[1:]:
        print(' + ' + line)
    print('')

    config.sliding_partner_maxdist = sim['sliding_partner_maxdist']
    config.n_threads = sim['n_threads']

    all_varnames = list(sim.keys())
    all_varvalues = list(sim.values())

    full_cline = ''
    for i in range(len(all_varnames)):
        full_cline += ' --' + all_varnames[i] + ' ' + str(all_varvalues[i])
    print(full_cline, file=open(sim['resultspath'] + sim['filelabel'] + '_cline.txt', 'w'))

    return sim


###############################################################################

    				### Values to typically leave unchanged ###



def args_to_dict(arg_names, args):
    """ Create a "sim" dictionary from command-line parameters """
    sim = {}
    full_cline = ''
    for arg_name in arg_names:
    	arg_val = vars(args)[arg_name]
    	if type(arg_val) == str:
    		arg_val_str = f'\'{arg_val}\''
    	else:
    		arg_val_str = f'{arg_val}'
    	set_arg_string = f'sim[\"{arg_name}\"] = ' + arg_val_str
    	exec(set_arg_string) # this is probably terrible programming practice
    	full_cline += (f'  --{arg_name} ' + arg_val_str)
    return sim, full_cline


def read_cl_record(fileprefix):
    """ For analysis.py funcctions, reload "sim" dictionary """
    with open(fileprefix + '_cline.txt','r') as file_in:
        cl_record = file_in.read()
    sim = string_to_dict(cl_record)

    # in case any keys are missing, fill with defaults
    default_sim, default_help = defaults.set_defaults()
    for key in default_sim.keys():
        if not key in sim.keys():
            sim[key] = default_sim[key]

    config.configure(sim)
    return sim


def string_to_dict(varstring):
    """ Transformation of saved sim text file back into a dictionary of
        various variable types
    """
    sim = {}
    spl = varstring.replace('\n','').split('--')
    names_and_vals = [ item.split(' ')[0:2] for item in spl ]

    for name_and_val in names_and_vals:
        varname = name_and_val[0]
        valstring = name_and_val[1]
        if valstring == 'True':
            val = True
        elif valstring == 'False':
            val = False
        elif sum([ int(item.isalpha() and item != 'e')  for item in valstring ]) > 0:
            # there is a letter
            val = valstring.replace("\'","")
        else:
            num_sides_of_decimal_point = sum(([ int(item.replace('-','').isnumeric()) for item in valstring.replace('e-','.').replace('e','.').split('.') ]))
            if num_sides_of_decimal_point == 1:
                val = int(valstring)
            elif num_sides_of_decimal_point in [2,3] :
                val = float(valstring)
            else:
                val = ""
        if varname != '':
            sim[varname] = val
    return sim

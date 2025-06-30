#!/usr/bin/env python3

#Order Parameter Analysis for ensemble

from analysis import *
import sys

#input from terminal/script
name = sys.argv[1] #path to the result file that needs to be analysed
save_path = sys.argv[2] #path to save the analysis file

# output file path and name can be set
write = "Chiral_analysis"
write_file = save_path + write
file1 = open(write_file, "w+")

#input file name
fileprefix = name + "chiral_test"  # 'chiral_test' referes to the filelable set in Chiral_parameters_script

#end time is obtained here
time_end = int(sorted(glob.glob(fileprefix + '*.dat'), key=os.path.getmtime)[-1].split('.dat')[0].split('_')[-1])


for j in range(1,time_end, 10):
    chains =  recall(filelabel = "chiral_test", filenumber = j, resultspath = name)
    pop = polar_orderparameter(chains)
    nop, ang = nematic_orderparameter(chains)
    time = chains['t']
    beads = chains["pops"]['beads']
    file1.write("%f %f %f %f\n" %(time, pop, nop, ang))

file1.close()


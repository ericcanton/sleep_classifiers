"""
The derivative of the magnitude of acceleration data is plotted,
along with a scatterplot of the human-labeled PSG data. 

-----------------------------------------------------------------

One should call this script from the directory 
    sleep_classifiers/source/
since the script, by default, assumes data is in 
    sleep_classifiers/data/
and so looks in the relative paths 
    ../data/motion  and  ../data/labels
This can be changed by passing -i option and providing a relative
or absolute path to the motion/ and labels/ directories; see below.

Output directories can be specified to change them
from the defaults: ../data/images and ../data/timestats 

-----------------------------------------------------------------

Usage (basic):
    $ python scatter.py 
This looks for acceleration data and PSG labels in 
    ../data/motion/  and  ../data/labels
Outputs the scatterplots to 
    ./data/images 
and summary statistics of the Apple Watch timestamps to 
    ./data/timestats

Optional arguments, can be passed in any order: 
    '-o'    Specify an output directory. Can be relative or absolute.
    '-i'    Specify an input directory 
            (where 'NNNN_acceleration.txt' files are stored)
    '-pos'  Only consider data with non-negative timestamps

-----------------------------------------------------------------

Example usage (full):
    $ python scatter.py -pos -o /home/ericcanton/sleep_outputs/ -i /home/ericcanton/sleep_inputs/
This command modifies the behavior of scatter.py to...
*) filter the acceleration data to only consider times >= 0.

*) look for the acceleration data in 
        /home/ericcanton/sleep_inputs/motion/
    and PSG labeling in 
        /home/ericcanton/sleep_inputs/labels/

*) write the scatterplots to 
        /home/ericcanton/sleep_outputs/images
    and the timestamp stats (count, mean, std, quartiles, etc) to 
        /home/ericcanton/sleep_outputs/timestats

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import glob # for iterating over files
from os import mkdir
import sys

#################################################################
# Helper functions for magnitude of 3d data, numerical derivative
#################################################################

def mag(x,y,z):
    return (x**2 + y**2 + z**2)**(0.5)
vmag = np.vectorize(mag)

def vder(f : pd.DataFrame):
    t0, t1 = f.iloc[:-1, 0], f.iloc[1:, 0]
    f0, f1 = f.iloc[:-1, 1], f.iloc[1:, 1]

    # t1 and f1 are indexed 1..n
    # change to match t0, f0 indexed 0..(n-1)
    t1.reset_index(drop=True, inplace=True)
    f1.reset_index(drop=True, inplace=True)

    # Calculate the derivative
    df = (f1 - f0)/(t1 - t0)

    return (t0, t1, df)


#################################################################
# Create directories to store summary stats and images
#################################################################
if '-o' in sys.argv:
    outpath = sys.argv[sys.argv.index('-o')+1]
    if outpath[-1] != '/':
        outpath += '/'
else:
    outpath = '../data/motion/'

try:
    mkdir(outpath + 'images')
    print("Images output folder created!")
except:
    print("Images output folder already exists.")

try:
    mkdir(outpath + 'timestats')
    print("Timestats output folder created!")
except:
    print("Timestats output folder already exists.")

print()

#################################################################
# Get a list of the files to be read. 
# Assumes these stored in ../data/motion
#################################################################

if '-i' in sys.argv:
    inpath = sys.argv[sys.argv.index('-i') + 1]
    if inpath[-1] != '/':
        inpath += "/"
else:
    inpath = '../data/'


for p in glob.glob(inpath + "motion/*.txt"):

    #############################################################
    # Extract the subject identifier. 
    # Filenames have the form NNNNNN_acceleration.txt
    #############################################################
    subject = p.split("/")[-1]
    subject = subject[:subject.find("_")]
    print("Working on {}...".format(subject))

    #############################################################
    # Load the dataset into a DataFrame
    #############################################################
    accel = pd.read_csv(p, sep=' ', names=['time', 'x', 'y', 'z'])

    if '-pos' in sys.argv:
        accel = accel[accel['time'] >= 0]
        accel.reset_index(drop=True, inplace=True)

    accel['Magnitude'] = vmag(accel['x'], accel['y'], accel['z'])

    psg = pd.read_csv("./data/labels/{}_labeled_sleep.txt".format(subject), 
            sep=' ', 
            names=['time', 'label'])

    # then find derivative
    t0, t1, df = vder(accel[['time', 'Magnitude']])

    #############################################################
    # Write summary statistics for time differences to file.
    #############################################################

    path = outpath + 'timestats/{}_timestats.txt'.format(subject)
    stats = '''
Summary statistics for timestamp differentials (i.e. t[i] - t[i-1]) from {}:

'''.format(subject)

    w = (t1-t0).describe() # type(w) == pd.core.series.Series

    # w.index == ['count', 'mean', 'std', 'min', '25%', '50%', ..]
    for i in w.index: 
        # append lines from w
        # format example: 
        #   count   123456
        #   mean    55.54301
        #   std     0.0123
        #   ...
        stats += '{}\t{}\n'.format(i, w[i])

    with open(path, 'w') as f:
        f.write(stats)
    print("Stats on timestamps written...")
    
    fig = plt.figure(figsize=(40,10), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.set_xlim(t0.min(), t1.max())
    ax.set_ylim(-11, 11)

    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xlabel("Time in seconds")

    plt.grid(b=True, axis='x')

    ax.scatter(t0, df, c='#00274C')
    ax.scatter(psg['time'], psg['label'], c='#FFCB05')

    plt.savefig(outpath + "images/{}.png".format(subject))
    print("Image created and saved.\n")

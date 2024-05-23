from __future__ import print_function
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout
import parmed as pmd
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as mdt
import os
import re
import numpy as np
from numpy import array
import collections
from collections import defaultdict
import pandas as pd
import glob
import itertools
from contact_map import ContactMap, ContactFrequency, ContactDifference, ResidueContactConcurrence, plot_concurrence
import pickle

# data about system
Protein = 'SETD2'
Peptide = 'H3K36'
sim_time = '100ns'
replicates = '10'

# want to subtract two data frames? - DataFrames must have same length
subtract = True
subtracting_peptide = 'ssK36'

# load pickle file
infile = open('path_to_pickle/df_contacts_{}_{}_{}x{}.pkl'.format(Protein, Peptide, replicates, sim_time), 'rb')
df = pickle.load(infile)
infile.close()
print('pickle loded')

if subtract == True:
    # load other data Frame
    infile = open('path_to_pickle/df_contacts_{}_{}_{}x{}.pkl'.format(Protein, subtracting_peptide, replicates, sim_time), 'rb')
    df_subt = pickle.load(infile)
    infile.close()
    df = df - df_subt.values
    print('subtracting:{}-{}...'.format(NHT, NHT_subt))

# replace all values in df in the range of (-0.5) - 0.5 with NaN
df_cut = df.where((df >= 0.05) | (df <= -0.05), np.nan)

# delete all rows with only NaN in it (remaining NaN will not appear in excel)
df_drop = df_cut.dropna(axis=0, how='all')


# convert dataframe to excel
if subtract == True:
    print('given xlsx is:{}-{}'.format(Peptide, subtracting_peptide))
    df_drop.to_excel('contacts_{}_{}-{}.xlsx'.format(Protein, Peptide, subtracting_peptide).format(), index_label='{}-{}'.format(Peptide, subtracting_peptide))
if subtract == False:
    print('given xlsx is:{}'.format(Peptide))
    df_drop.to_excel('contacts_{}_{}_{}x{}.xlsx'.format(Protein, Peptide, replicates, sim_time).format(), index_label='{}'.format(Peptide))

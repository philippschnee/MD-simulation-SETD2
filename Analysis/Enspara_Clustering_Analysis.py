from __future__ import print_function
import enspara
from enspara import ra
import pickle
import mdtraj as mdt
import numpy as np
import matplotlib.pyplot as plt
import itertools
import more_itertools as mit
import pandas as pd
import os

cluster_centers = 2
Replicate = '1'

create_centroid_pdb = False

folder = 'name_of_folder'
os.system('mkdir {0}'.format(folder))

cl_ass=enspara.ra.load('fs-khybrid-clusters0020-assignments.h5')
cl_ass_list=np.ndarray.tolist(cl_ass)
cl_ass_list_flat= [val for sublist in cl_ass_list for val in sublist]

d={}
for i in range (cluster_centers):
    d['centroid_{}'.format(i)] = cl_ass_list_flat.count(i)

df=pd.DataFrame(cl_ass)
H3K36=df.iloc[0:50]
ssK36=df.iloc[50:100]
count_H3K36=pd.value_counts(H3K36.values.ravel())
print('H3K36:',count_H3K36)
count_ssK36=pd.value_counts(ssK36.values.ravel())
print('ssK36:',count_ssK36)

#create and save a centroid pdb
if create_centroid_pdb == True:
    infile = open('fs-khybrid-clusters0020-centers.pickle', 'rb')
    new_file = pickle.load(infile)
    first_tr = new_file[0]
    first_tr.save_pdb(folder + '/' + '/centroid_0_Rep{}.pdb'.format(Replicate), 'w')
    sec_tr = new_file[1]
    sec_tr.save_pdb(folder + '/' + '/centroid_1_Rep{}.pdb'.format(Replicate), 'w')
    thr_tr = new_file[2]
    thr_tr.save_pdb(folder + '/' + '/centroid_2_Rep{}.pdb'.format(Replicate), 'w')

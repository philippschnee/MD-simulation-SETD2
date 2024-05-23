from __future__ import print_function
import mdtraj as mdt
import os, shutil
import re
import numpy as np
from numpy import array
import collections
from collections import defaultdict
import pandas as pd
import itertools
import warnings

warnings.filterwarnings("ignore", message="top= kwargs ignored since this file parser does not support it")


peptide = 'H3K36' #or ssK36

get_rmsd = True
get_End_to_End = True
get_SN2 = False

list_NAC = []
list_angle_109 = []
list_angle_linear = []
result_min_rmsd = []
frame_min_rmsd = []
traj_list = []
suc_docking_traj = []
suc_docking_NAC = {}
suc_docking_rmsd = {}

df_rmsd_ges = pd.DataFrame()

# load trajectories
spec_traj = []          #put here replicate number of simulation  you want analyse e.g. replicate [1, 3, 4]
for i in spec_traj:
    traj = mdt.load('/path_to_trajectory/production_{}.h5'.format(i))
    topology =traj.topology
    
    # load reference trajectory for rmsd calcualtion 
    if peptide == 'H3K36':
         ref = mdt.load('/path_to_trajectory/production_{}.h5')
    if peptide == 'ssK36':
         ref = mdt.load('/path_to_trajectory/production_{}.h5')
    ref_top =ref.topology
    
    # select only peptides for rmsd analysis
    sel_pep = topology.select('chainid 1 and element != "H"')           #chainid 1 = peptide
    ref_sel_pep = ref_top.select('chainid 1 and element != "H"')
    pep = traj.atom_slice(sel_pep)
    ref_pep = ref.atom_slice(ref_sel_pep)
    
    #calculate rmsd and put all values in list
    rmsd = mdt.rmsd(pep, ref_pep, frame=1)
    rmsd = rmsd.tolist()
    
    #get every 10th rmsd value -> to minimize storage stress
    if get_rmsd == True:
         rmsd_short = rmsd[::10]
   
    #get every 10th End-to-End distance
    if get_End_to_End == True:
         table, bonds = topology.to_dataframe()
         df = pd.DataFrame(table)
         
         df_search = df[(df['chainID'] == 1) & (df['resName'] == 'ALA') & (df['resSeq'] == 29) & (df['name'] == 'N')]
         A29_N = df_search.index.item()
         df_search = df[(df['chainID'] == 1) & (df['resName'] == 'PRO') & (df['resSeq'] == 43) & (df['name'] == 'N')]
         P43_N = df_search.index.item()
         df_search = df[(df['chainID'] == 2) & (df['resName'] == 'SAM') & (df['resSeq'] == 1804) & (df['name'] == 'N')]
         K36_NZ = df_search.index.item()
         
         End_to_End = [[A29_N,P43_N], [A29_N,K36_NZ]]
         End_to_End_dist = mdt.compute_distances(traj,End_to_End, periodic=True, opt=True)
         dist_End_to_end = np.ndarray.tolist(End_to_End_dist)
         list_dist_End_to_End_merged = list(itertools.chain.from_iterable(dist_End_to_end))       
         C_to_N = list_dist_End_to_End_merged[::2]
         C_to_N_short = C_to_N[::10]

         array = np.column_stack([rmsd_short,  C_to_N_short])
         df_rmsd = pd.DataFrame(array)
         df_rmsd.columns = ['rmsd','C_to_N']
         df_rmsd_ges = pd.concat([df_rmsd_ges,df_rmsd], axis = 1)
         
    #calculate SN2 criteria
    if get_SN2 == True:
         table, bonds = topology.to_dataframe()
         df = pd.DataFrame(table)
         
         df_search = df[(df['chainID'] == 1) & (df['resName'] == 'LYS') & (df['resSeq'] == 36) & (df['name'] == 'CE')]
         K36_CE = df_search.index.item()
         df_search = df[(df['chainID'] == 1) & (df['resName'] == 'LYS') & (df['resSeq'] == 36) & (df['name'] == 'NZ')]
         K36_NZ = df_search.index.item()
         df_search = df[(df['chainID'] == 2) & (df['resName'] == 'SAM') & (df['resSeq'] == 1804) & (df['name'] == 'N')]
         SAM_N = df_search.index.item()
         df_search = df[(df['chainID'] == 2) & (df['resName'] == 'SAM') & (df['resSeq'] == 1804) & (df['name'] == 'SD')]
         SAM_SD = df_search.index.item()
         df_search = df[(df['chainID'] == 2) & (df['resName'] == 'SAM') & (df['resSeq'] == 1804) & (df['name'] == 'CE')]
         SAM_CE = df_search.index.item()
         
         SETD2_dist = [[K36_NZ,SAM_CE],[K36_NZ,SAM_N]]
         SETD2_109 = [[K36_CE, K36_NZ, SAM_CE]]
         SETD2_linear = [[SAM_SD, SAM_CE, K36_NZ]]
         
         dist_NAC = mdt.compute_distances(traj,SETD2_dist, periodic=True, opt=True)
         angle_109 = mdt.compute_angles(traj, array(SETD2_109), periodic=True, opt=True)
         angle_linear = mdt.compute_angles(traj, array(SETD2_linear), periodic=True, opt=True)
     
         
         #write all distances in a list but only from first atom pair
         dist_NAC_list = np.ndarray.tolist(dist_NAC)
         list_dist_NAC_merged = list(itertools.chain.from_iterable(dist_NAC_list))
         NAC = list_dist_NAC_merged[::2]
         list_NAC.append(NAC)
         
         #tranform all dihedrals (BB, SC) from rad in degrees and write them in a list
         angle_109=np.rad2deg(angle_109)
         angle_linear=np.rad2deg(angle_linear)
         
         angle_109_list=np.ndarray.tolist(angle_109)
         angle_linear_list=np.ndarray.tolist(angle_linear)
         
         angle_109_list_merged = list(itertools.chain.from_iterable(angle_109_list))
         angle_linear_list_merged = list(itertools.chain.from_iterable(angle_linear_list))
         
         list_angle_109.append(angle_109_list_merged)
         list_angle_linear.append(angle_linear_list_merged)
         
         #criteria lists in array
         criteria_array = np.column_stack([NAC,  angle_linear_list_merged, angle_109_list_merged, rmsd])
         df2 = pd.DataFrame(criteria_array)
         df2.columns = ['NAC','linear','109','rmsd']
         df_sliced = df2[(df2['NAC'] <0.4) & (df2['linear'] >150) & (df2['linear'] <210) & (df2['109'] >79) & (df2['109'] <139)] #+-30°
    
         # check for non-suc-docking simulations
         if not df_sliced.empty:
                 list_NAC = list(df2['NAC'])
                 suc_docking_NAC.update({i:list_NAC})
    
         if not df_sliced.empty:
                 list_rmsd = list(df2['rmsd'])
                 suc_docking_rmsd.update({i:list_rmsd})
    
         # check for non-suc-docking simulations
         df_dummy = pd.DataFrame({'NAC':[1], 'linear':[1], '109':[1], 'rmsd':[1]})
    
         if df_sliced.empty:
                 df_sliced = pd.concat([df_sliced,df_dummy],ignore_index=True)
    
         # select frame with smallest rmsd
         min_rmsd = df_sliced[df_sliced.rmsd == df_sliced.rmsd.min()]
         frame = min_rmsd.index.tolist()
    
         if len(frame)>0:
                 traj = traj.slice(frame)
                 traj_list.append(traj)
    
         if len(frame)<1:
                 frame.append(0)
    
         result_min_rmsd.append(min_rmsd.iloc[0]['rmsd'])
         frame_min_rmsd.append(frame[0])

         #results in excel
         dict_for_pd = {'min RMSD':result_min_rmsd,'frame':frame_min_rmsd}
         df3 = pd.DataFrame(dict_for_pd)
         df3.to_excel(('results sMD unfolding {}_Replicate{}.xlsx').format(peptide, i))

print(df_rmsd_ges)
df_rmsd_ges.to_excel('RMSD_End_Distance_Correaltion_{}.xlsx'.format(peptide))
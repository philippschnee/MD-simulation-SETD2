from __future__ import print_function
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

### after this script, let run "Analysis_SETD2-Contact_Maps_Analysis" ###

# input parameters
Protein = 'SETD2'
Peptide = 'H3K36' #or ssK36
sim_time = '100ns'

show_SN2 = True

use_all = True
use_only_SN2 = False
use_only_NON_SN2 = False

number_replicates = 1
replicates = '1'        #put in same as number_replicates


# load trajectory and topology
traj_dict = {}
for i in range(number_replicates):
    folder = '/path_to_trajectory/production_SETD2_H3K36_100ns_{}.h5'.format(i+1)
    print(folder)
    traj_dict[i+1]=mdt.load(folder)

traj_list = []
for key in traj_dict:
    traj_list.append(traj_dict[key])

traj_pre = mdt.join(traj_list,check_topology=True, discard_overlapping_frames=True)
traj = traj_pre.superpose(traj_pre,frame=0,parallel=True)
print('Trajectories successfully loaded, joined and superposed')

topology = traj.topology


# calculate SN2 criteria
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

SAM_dist = [[K36_NZ,SAM_CE],[K36_NZ,SAM_N]]
SAM_109 = [[K36_CE, K36_NZ, SAM_CE]]
SAM_linear = [[SAM_SD, SAM_CE, K36_NZ]]


dist_=mdt.compute_distances(traj,SAM_dist, periodic=True, opt=True)
angle_109 = mdt.compute_angles(traj, array(SAM_109), periodic=True, opt=True)
angle_linear = mdt.compute_angles(traj, array(SAM_linear), periodic=True, opt=True)

#write all distances in a list but only from first atom pair
list_dist = []
list_angle_109 = []
list_angle_linear = []
traj_list = []

dist_list = np.ndarray.tolist(dist_)
list_dist_merged = list(itertools.chain.from_iterable(dist_list))

distance = list_dist_merged[::2]
list_dist.append(dist_)

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
criteria_array = np.column_stack([distance,  angle_linear_list_merged, angle_109_list_merged])
df = pd.DataFrame(criteria_array)
df_sliced = df[(df[0] <0.4) & (df[1] >150) & (df[1] <210) & (df[2] >79) & (df[2] <139)] #+-30°, 4.0 A distance NZ-CE, 150°-210° angle_linear, 79°-139° angle_109

indices = df_sliced.index.array
indices_df = pd.DataFrame(indices)
temp = indices_df.values.tolist()
INT = list(itertools.chain.from_iterable(temp))

print('total number of SN2 frames in SETD2 {}:'.format(Peptide), len(INT))


# use only SN2 frames for contact calculation
if use_only_SN2 == True:
 traj = traj.slice(INT)
  

# use only NON SN2 frames for contact calculation
if use_only_NON_SN2 == True:
 non_SN2_frames = list(range(0,50000)) #number of frames in a 100 ns simulation, saving 10000 steps
 for i in non_SN2_frames[:]:
    if i in INT:
        non_SN2_frames.remove(i)
 traj = traj.slice(non_SN2_frames)

# select peptide and protein
peptide = topology.select('chainid 1 and element != "H"')
protein = topology.select('chainid 0 and element != "H"')

# resid -1 from pdb
print('Calculating Frequency of contacts')
contacts = ContactFrequency(traj, query=peptide, haystack=protein, cutoff=0.45)
x = contacts.residue_contacts.sparse_matrix.toarray()
df = pd.DataFrame(x)

df.drop(df.iloc[:, 0:255], inplace = True, axis = 1)
df.drop(df.columns[[15, 16, 17, 18]], axis = 1, inplace = True)
df.drop(df.index[254:273],0,inplace=True)
df.index = ['C1449',  'V1450',  'M1451',  'D1452',  'D1453',  'F1454',  'R1455',  'D1456',  'P1457',  'Q1458',  'R1459',  'W1460',  'K1461',  'E1462',  'C1463',  'A1464',  'K1465',  'Q1466',  'G1467',  'K1468',  'M1469',  'P1470',  'C1471',  'Y1472',  'F1473',  'D1474',  'L1475',  'I1476',  'E1477',  'E1478',  'N1479',  'V1480',  'Y1481',  'L1482',  'T1483',  'E1484',  'R1485',  'K1486',  'K1487',  'N1488',  'K1489',  'S1490',  'H1491',  'R1492',  'D1493',  'I1494',  'K1495',  'R1496',  'M1497',  'Q1498',  'C1499',  'E1500',  'C1501',  'T1502',  'P1503',  'L1504',  'S1505',  'K1506',  'D1507',  'E1508',  'R1509',  'A1510',  'Q1511',  'G1512',  'E1513',  'I1514',  'A1515',  'C1516',  'G1517',  'E1518',  'D1519',  'C1520',  'L1521',  'N1522',  'R1523',  'L1524',  'L1525',  'M1526',  'I1527',  'E1528',  'C1529',  'S1530',  'S1531',  'R1532',  'C1533',  'P1534',  'N1535',  'G1536',  'D1537',  'Y1538',  'C1539',  'S1540',  'N1541',  'R1542',  'R1543',  'F1544',  'Q1545',  'R1546',  'K1547',  'Q1548',  'H1549',  'A1550',  'D1551',  'V1552',  'E1553',  'V1554',  'I1555',  'L1556',  'T1557',  'E1558',  'K1559',  'K1560',  'G1561',  'W1562',  'G1563',  'L1564',  'R1565',  'A1566',  'A1567',  'K1568',  'D1569',  'L1570',  'P1571',  'S1572',  'N1573',  'T1574',  'F1575',  'V1576',  'L1577',  'E1578',  'Y1579',  'C1580',  'G1581',  'E1582',  'V1583',  'L1584',  'D1585',  'H1586',  'K1587',  'E1588',  'F1589',  'K1590',  'A1591',  'R1592',  'V1593',  'K1594',  'E1595',  'Y1596',  'A1597',  'R1598',  'N1599',  'K1600',  'N1601',  'I1602',  'H1603',  'Y1604',  'Y1605',  'F1606',  'M1607',  'A1608',  'L1609',  'K1610',  'N1611',  'D1612',  'E1613',  'I1614',  'I1615',  'D1616',  'A1617',  'T1618',  'Q1619',  'K1620',  'G1621',  'N1622',  'C1623',  'S1624',  'R1625',  'F1626',  'M1627',  'N1628',  'H1629',  'S1630',  'C1631',  'E1632',  'P1633',  'N1634',  'C1635',  'E1636',  'T1637',  'Q1638',  'K1639',  'W1640',  'T1641',  'V1642',  'N1643',  'G1644',  'Q1645',  'L1646',  'R1647',  'V1648',  'G1649',  'F1650',  'F1651',  'T1652',  'T1653',  'K1654',  'L1655',  'V1656',  'P1657',  'S1658',  'G1659',  'S1660',  'E1661',  'L1662',  'T1663',  'F1664',  'D1665',  'Y1666',  'Q1667',  'F1668',  'Q1669',  'R1670',  'Y1671',  'G1672',  'K1673',  'E1674',  'A1675',  'Q1676',  'K1677',  'C1678',  'F1679',  'C1680',  'G1681',  'S1682',  'A1683',  'N1684',  'C1685',  'R1686',  'G1687',  'Y1688',  'L1689',  'G1690',  'G1691',  'E1692',  'N1693',  'R1694',  'V1695',  'S1696',  'I1697',  'R1698',  'A1699',  'A1700',  'G1701',  'G1702',  'K1703']
df.columns = ['A29','P30','A31','T32','G33','G34','F35','K36','K37','P38','H39','R40','Y41','R42','P43']

# create new folder and safe pickle, excel sheet
if use_all == True:
 new_folder = 'contacts_{}_{}_{}x{}'.format(Protein, Peptide, replicates, sim_time)
if use_only_SN2 == True:
 new_folder = 'contacts_{}_{}_{}x{}_SN2'.format(Protein, Peptide, replicates, sim_time)
if use_only_NON_SN2 == True:
 new_folder = 'contacts_{}_{}_{}x{}_NON'.format(Protein, Peptide, replicates, sim_time)

os.system('mkdir {0}'.format(new_folder))
df.to_pickle(open(new_folder + '/' +  'df_contacts_{}_{}_{}x{}.pkl'.format(Protein, Peptide, replicates, sim_time), 'wb'))
df.to_excel(open(new_folder + '/' +  'contacts_{}_{}_{}x{}.xlsx'.format(Protein, Peptide, replicates, sim_time), 'wb'))

### after this script, let run "Analysis_Contact_Maps_Analysis" ###
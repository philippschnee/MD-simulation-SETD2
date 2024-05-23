from __future__ import print_function
from openmm.app import *
from openmm import *
import openmm as mm
from openmm.unit import *
from sys import stdout
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as mdt
import os
import re
import numpy as np


peptide = 'ssK36'
sim_time = '100ns'
traj_folder = 'SETD2_{}_{}_complex'.format(peptide, sim_time)
number_replicates = 1

# Input Files
pdb = PDBFile('SETD2_{}_complex.pdb'.format(peptide))
protein = app.Modeller(pdb.topology, pdb.positions)
sim_forcefield = ('amber14-all.xml')
sim_watermodel = ('amber14/tip4pew.xml')
sim_gaff = 'gaff.xml'

ligand_names = ['SAM', 'ZNB']                                                #list of ligand names: ['LIGANDNAME1','LIGANDNAME2', ...]
ligand_xml_files = ['SAM.xml', 'ZNB.xml']                                    #list of ligand parameter xml files: ['LIGANDNAME1.xml','LIGANDNAME2.xml', ...]
ligand_pdb_files = ['SAM.pdb', 'ZNB.pdb']                                    #list of ligand pdb files: ['LIGANDNAME1.pdb','LIGANDNAME2.pdb', ...]

# Integration Options
dt = 0.002*picoseconds
temperature = 300*kelvin
friction = 1/picosecond
sim_ph = 7.0

# Simulation Options

Simulate_Steps = 50000000              # 100 ns
npt_eq_Steps = 2500000                 # 5 ns
SAM_restr_eq_Steps = 2500000           # 5 ns
SAM_free_eq_Steps = 2500000            # 5 ns

restrained_eq_atoms = 'protein and name CA and not chainid 1'                #MDTraj selection syntax; restrained backbone npt/nvt_equilibration_steps
force_eq_atoms = 100                                                         #kilojoules_per_mole/unit.angstroms

restrained_eq_atoms2 = 'chainid 3 and resid 0'                               # MDTraj selection syntax; restrained peptide, cofactor, and metal-dummy atoms during npt_ligand_restrained_equilibration_steps
force_eq_atoms2 = 150 

restrained_ligands = True #(TRUE|FALSE)                                      # restrain ligands for protein only equilibration (npt_ligand_restrained_equilibration_steps)

platform = Platform.getPlatformByName('CUDA')
gpu_index = '0'
platformProperties = {'Precision': 'single','DeviceIndex': gpu_index}
trajectory_out_atoms = 'protein or resname SAM or resname ZNB'
trajectory_out_interval = 10000

# Protonations are given as dictionary: Key is a tuple (chain name, residue number), value is a protonation state Histidine: HIE, HID, HIP, HIN; Glutamate: GLU, GLH; Aspartate: ASP, ASH; Lysine: LYN, LYS; Cysteine: CYS, CYX
# example: protonation_dict = {('A',84): 'HIP', ('A',86): 'HID'}

protonation_dict = {('A',1499): 'CYX', ('A',1501): 'CYX', ('A',1516):'CYX', ('A',1520): 'CYX', ('A',1529): 'CYX', ('A',1533):'CYX', ('A',1539):'CYX', ('A',1631):'CYX', ('A',1678):'CYX', ('A',1680):'CYX', ('A',1685):'CYX', ('B',36):'LYN'} #only for manual protonation


# Prepare the Simulation
os.system('mkdir {0}'.format(traj_folder))

# build force field object | all xml parameter files should be bundled here
xml_list = [sim_forcefield, sim_gaff, sim_watermodel]
for lig_xml_file in ligand_xml_files:
	xml_list.append(lig_xml_file)
forcefield = app.ForceField(*xml_list)

# protonate protein and also assign custom protonations defined above
protonation_list = []
key_list=[]

if len(protonation_dict.keys()) > 0:
    for chain in protein.topology.chains():
        chain_id = chain.id
        
        protonations_in_chain_dict = {}
        for protonation_tuple in protonation_dict:
            if chain_id == protonation_tuple[0]:
                residue_number = protonation_tuple[1]
                protonations_in_chain_dict[int(residue_number)] = protonation_dict[protonation_tuple]
                key_list.append(int(residue_number))                                               
        
        for residue in chain.residues():
            with open("log_prot.txt", "a") as myfile:
                residue_id = residue.id
                myfile.write(residue_id)
                if int(residue_id) in key_list:
                    myfile.write(': Protoniert!')
                    myfile.write(residue_id)
                    protonation_list.append(protonations_in_chain_dict[int(residue_id)])
                else:
                    protonation_list.append(None)
                    myfile.write('-')          
              
protein.addHydrogens(forcefield, pH=sim_ph, variants = protonation_list)

# add ligand structures to the model
for lig_pdb_file in ligand_pdb_files:
	ligand_pdb = app.PDBFile(lig_pdb_file)
	protein.add(ligand_pdb.topology, ligand_pdb.positions)

# Generation and Solvation of Box
print('Generation and Solvation of Box')
boxtype = 'cubic' #('cubic'|'rectangular')
box_padding = 1.0 #nanometers
x_list = []
y_list = []
z_list = []

# get atom indices for protein plus ligands
for index in range(len(protein.positions)):
	x_list.append(protein.positions[index][0]._value)
	y_list.append(protein.positions[index][1]._value)
	z_list.append(protein.positions[index][2]._value)
x_span = (max(x_list) - min(x_list))
y_span = (max(y_list) - min(y_list))
z_span = (max(z_list) - min(z_list))

# build box and add solvent
d =  max(x_span, y_span, z_span) + (2 * box_padding)

d_x = x_span + (2 * box_padding)
d_y = y_span + (2 * box_padding)
d_z = z_span + (2 * box_padding)

prot_x_mid = min(x_list) + (0.5 * x_span)
prot_y_mid = min(y_list) + (0.5 * y_span)
prot_z_mid = min(z_list) + (0.5 * z_span)

box_x_mid = d_x * 0.5
box_y_mid = d_y * 0.5
box_z_mid = d_z * 0.5

shift_x = box_x_mid - prot_x_mid
shift_y = box_y_mid - prot_y_mid
shift_z = box_z_mid - prot_z_mid

solvated_protein = app.Modeller(protein.topology, protein.positions)

# shift coordinates to the middle of the box
for index in range(len(solvated_protein.positions)):
	solvated_protein.positions[index] = (solvated_protein.positions[index][0]._value + shift_x, solvated_protein.positions[index][1]._value + shift_y, solvated_protein.positions[index][2]._value + shift_z)*nanometers

# add box vectors and solvate
if boxtype == 'cubic':
	solvated_protein.addSolvent(forcefield, model='tip4pew', neutralize=True, ionicStrength=0.1*molar, boxVectors=(mm.Vec3(d, 0., 0.), mm.Vec3(0., d, 0.), mm.Vec3(0, 0, d)))
elif boxtype == 'rectangular':
	solvated_protein.addSolvent(forcefield, model='tip4pew', neutralize=True, ionicStrength=0.1*molar, boxVectors=(mm.Vec3(d_x, 0., 0.), mm.Vec3(0., d_y, 0.), mm.Vec3(0, 0, d_z)))


# Building System

print('Building system...')
topology = solvated_protein.topology
positions = solvated_protein.positions
selection_reference_topology = mdt.Topology().from_openmm(solvated_protein.topology)
trajectory_out_indices = selection_reference_topology.select(trajectory_out_atoms)
restrained_eq_indices = selection_reference_topology.select(restrained_eq_atoms)
restrained_eq_indices2 = selection_reference_topology.select(restrained_eq_atoms2)
system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers,ewaldErrorTolerance=0.0005, constraints=HBonds, rigidWater=True)
integrator = LangevinIntegrator(temperature, friction, dt)
simulation = Simulation(topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(positions)


# Minimize

print('Performing energy minimization...')
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'PreMin.pdb', 'w'), keepIds=True)
simulation.minimizeEnergy()
min_pos = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'PostMin.pdb', 'w'), keepIds=True)
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
print('System is now minimized')


# Restraints

force = mm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
force.addGlobalParameter("k", force_eq_atoms*kilojoules_per_mole/angstroms**2)
force.addPerParticleParameter("x0")
force.addPerParticleParameter("y0")
force.addPerParticleParameter("z0")

force2 = mm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
force2.addGlobalParameter("k", force_eq_atoms2*kilojoules_per_mole/angstroms**2)
force2.addPerParticleParameter("x0")
force2.addPerParticleParameter("y0")
force2.addPerParticleParameter("z0")


if restrained_ligands:
    for res_atom_index in restrained_eq_indices2:
        force2.addParticle(int(res_atom_index), min_pos[int(res_atom_index)].value_in_unit(nanometers))
    system.addForce(force2)

for res_atom_index in restrained_eq_indices:
	force.addParticle(int(res_atom_index), min_pos[int(res_atom_index)].value_in_unit(nanometers))
system.addForce(force)

# NPT Equilibration

# add barostat for NPT
system.addForce(mm.MonteCarloBarostat(1*atmospheres, temperature, 25))
simulation.context.setPositions(min_pos)
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=npt_eq_Steps, separator='\t'))
simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT.h5', 10000, atomSubset=trajectory_out_indices))
print('restrained NPT equilibration...')
simulation.step(npt_eq_Steps)
state_npt_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
positions = state_npt_EQ.getPositions()
app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'post_NPT_EQ.pdb', 'w'), keepIds=True)
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
print('Successful NPT equilibration!')


# Free Equilibration
# forces: 0->HarmonicBondForce, 1->HarmonicAngleForce, 2->PeriodicTorsionForce, 3->NonbondedForce, 4->CMMotionRemover, 5->CustomExternalForce, 6->CustomExternalForce, 7->MonteCarloBarostat
n_forces = len(system.getForces())
system.removeForce(n_forces-2)
print('force removed')

# optional ligand restraint to force slight conformational changes
if restrained_ligands:
    
    integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.002*picoseconds)
    simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
    simulation.context.setState(state_npt_EQ)
    simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_restr_eq_Steps, separator='\t'))
    simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'free_BB_restrained_SAM_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
    print('free BB NPT equilibration of protein with restrained SAM...')
    simulation.step(SAM_restr_eq_Steps)
    state_free_EQP = simulation.context.getState(getPositions=True, getVelocities=True)
    positions = state_free_EQP.getPositions()
    app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_BB_restrained_SAM_NPT_EQ.pdb', 'w'), keepIds=True)
    print('Successful free BB, SAM restrained equilibration!')
  
    # equilibration with free ligand   
    n_forces = len(system.getForces())
    system.removeForce(n_forces-2)
    integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.002*picoseconds)
    simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
    simulation.context.setState(state_free_EQP)
    simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_free_eq_Steps, separator='\t'))
    simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'SAM_free_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
    print('SAM free NPT equilibration...')
    simulation.step(SAM_free_eq_Steps)
    state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
    positions = state_free_EQ.getPositions()
    app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'SAM_free_NPT_EQ.pdb', 'w'), keepIds=True)
    print('Successful SAM free equilibration!')
    
else:
    
    # remove ligand restraints for free equilibration (remove the second last force object, as the last one was the barostat)
    n_forces = len(system.getForces())
    system.removeForce(n_forces-2)
    simulation.context.setState(state_npt_EQ)
    simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_free_eq_Steps, separator='\t'))
    simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT_free.h5', 10000, atomSubset=trajectory_out_indices))
    print('free NPT equilibration...')
    simulation.step(SAM_free_eq_Steps)
    state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
    positions = state_free_EQ.getPositions()
    app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_NPT_EQ.pdb', 'w'), keepIds=True)
    print('Successful free equilibration!')


count = 1
while (count <= number_replicates):
 
 # Simulate 
 print('Simulating...')
  
 # create new simulation object for production run with new integrator
 integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.002*picoseconds)
 simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
 simulation.context.setState(state_free_EQ)
 simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=Simulate_Steps, separator='\t'))
 simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'production_SETD2_{}_{}_complex_{}.h5'.format(peptide, sim_time, count), 10000, atomSubset=trajectory_out_indices))
 print('production run of replicate {}...'.format(count))
 simulation.step(Simulate_Steps)
 state_production = simulation.context.getState(getPositions=True, getVelocities=True)
 state_production = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
 final_pos = state_production.getPositions()
 app.PDBFile.writeFile(simulation.topology, final_pos, open(traj_folder + '/' + 'production_SETD2_{}_{}_complex_{}.pdb'.format(peptide, sim_time, count), 'w'), keepIds=True)
 print('Successful production of replicate {}...'.format(count))
 del(simulation)
 count = count+1

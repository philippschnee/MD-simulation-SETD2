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

# Input parameters. Needed to find and name the files.

peptide = 'ssK36'								# name of the peptide
traj_folder = 'trajectories_{}'.format(peptide)	# name of the trajectory folder
number_replicates = 50							# number of simulated replicates

# simulation 

count = 0
while (count < number_replicates):
 count = count+1
 
 # input Files
 pdb = PDBFile('Production_peptide_simulation_{}.pdb'.format(peptide))		# this is your starting strcture of the peptide
 protein = app.Modeller(pdb.topology, pdb.positions)
 sim_forcefield = ('amber14-all.xml')			# this is the used forcefield
 sim_watermodel = ('amber14/tip4pew.xml')		# this is the used water model
 sim_gaff = 'gaff.xml'
 
 # integration Options
 dt = 0.002*picoseconds
 temperature = 353.15*kelvin # 5°C
 friction = 1/picosecond
 sim_ph = 7.0
 
 # simulation Options
 
 Simulate_Steps = 35000000               # 70 ns, simulation time of the main simualtion   
 
 npt_eq_Steps = 2500000                  # 5 ns, simulation time of the NPT equilibration
 free_eq_Steps = 2500000                 # 5 ns, simulation time of the unrestrained equilibration
 
 # information for the hardware you are simulating on. These parameters are deigned for NVIDIA GPUs
 platform = Platform.getPlatformByName('CUDA')
 gpu_index = '0'	# GPUs can be enumerated (0, 1, 2), if you don't know how this is set up, stick to 0
 platformProperties = {'Precision': 'single','DeviceIndex': gpu_index}
 trajectory_out_atoms = 'protein'
 trajectory_out_interval = 10000
 
 # prepare the Simulation
 
 os.system('mkdir {0}'.format(traj_folder))
 
 # build force field object | all xml parameter files should be bundled here
 xml_list = [sim_forcefield, sim_gaff, sim_watermodel]
 forcefield = app.ForceField(*xml_list)
 
 # Generation and Solvation of Box
 print('Generation and Solvation of Box')
 boxtype = 'cubic' # cubic or rectangular
 box_padding = 1.0 # in nanometers. This defines the distance of the protein to the box edges
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
 
 # NPT Equilibration
 
 # add barostat for NPT
 system.addForce(mm.MonteCarloBarostat(1*atmospheres, temperature, 25))
 simulation.context.setPositions(min_pos)
 simulation.context.setVelocitiesToTemperature(temperature*kelvin)
 simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=npt_eq_Steps, separator='\t'))
 print('restrained NPT equilibration...')
 simulation.step(npt_eq_Steps)
 state_npt_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
 positions = state_npt_EQ.getPositions()
 print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
 print('Successful NPT equilibration!')
 
 
 # Free Equilibration 
 # remove barostat for free equilibration (remove the second last force object, as the last one was the barostat)
 n_forces = len(system.getForces())
 system.removeForce(n_forces-1)
 simulation.context.setState(state_npt_EQ)
 simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=free_eq_Steps, separator='\t'))
 print('free NPT equilibration...')
 simulation.step(free_eq_Steps)
 state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
 positions = state_free_EQ.getPositions()
 print('Successful free equilibration!')
 
 
 # Simulate
 print('Simulating...')
 
 # create new simulation object for production run with new integrator
 integrator = mm.LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
 simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
 simulation.context.setState(state_free_EQ)
 simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=Simulate_Steps, separator='\t'))
 simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'production_{}_70ns{}.h5'.format(peptide, count), 10000, atomSubset=trajectory_out_indices))
 print('production run of replicate {}...'.format(count))
 simulation.step(Simulate_Steps)
 state_production = simulation.context.getState(getPositions=True, getVelocities=True)
 state_production = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
 final_pos = state_production.getPositions()
 app.PDBFile.writeFile(simulation.topology, final_pos, open(traj_folder + '/' + 'production_{}_70ns{}.pdb'.format(peptide, count), 'w'), keepIds=True)
 print('Successful production of replicate {}...'.format(count))
 del(simulation)

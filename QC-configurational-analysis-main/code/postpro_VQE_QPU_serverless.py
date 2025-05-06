# -*- coding: utf-8 -*-
"""

Processing of raw results from VQE_QPU.py

"""

import qiskit_optimization
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz, RealAmplitudes
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Sampler, Session, Options
# import qiskit_ibm_runtime

from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.vis import structure_vtk

from quantum_computing_functions import *
from quantum_computing_postprocessing import *

import qiskit as qiskit
import numpy as np
import pandas as pd
import timeit

from qiskit_algorithms.optimizers import SPSA, SLSQP, COBYLA

import matplotlib.pyplot as plt

from itertools import product
from datetime import timedelta

import copy

# import datetime
import time
import json
import re
import ast
from datetime import datetime

from qiskit_optimization.translators import from_ising, to_ising
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

plt.rcParams.update({'font.size': 25})
plt.rc('xtick', labelsize=25) 
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

current_datetime = datetime.now()
# Set the date and time of the files you want to call
timestr = '20250218-092804' #time.strftime("%Y%m%d-%H%M%S")

sim = 'N' # Y N

# Problem
n_supercell = 3
num_vac = 3

# Hyperparams
lambda_1 = 3
shots = 10000 
maxiters = 250 
tol = 1

optimiser_name = 'cobyla' 
obj_fun = 'CVAR' #'EV' 'CVAR', 'ACVAR' 
ansatz_name = 'RealAmp' # QAOA HEA RealAmp QAOAmixer
p = 2

alpha = 0.4
acvar_min = 0.2
acvar_max = 0.7

repeats = 1 


if sim == 'Y':
    sim_method = 'statevector' #'density_matrix',"statevector","matrix_product_state",extended_stabilizer
    # tensor_network
    device = 'stV'
    backend_name = 'stV'
elif sim == 'Y_noisy':
    instance = ''
    # To run on hardware, select the backend with the fewest number of jobs in the queue
    service = QiskitRuntimeService(channel="ibm_quantum", token="",
                                  instance=instance) # 
    # backend = service.least_busy(operational=True, simulator=False)
    real_backend = service.backend("ibm_fez")
    
    # try
    noise_model = NoiseModel.from_backend(real_backend)
    backend = AerSimulator(method='matrix_product_state', # matrix_product_state
                        noise_model=noise_model)
    
    device = 'Fake '+real_backend.name
    backend_name = device
else:
    instance = ''
    # To run on hardware, select the backend with the fewest number of jobs in the queue
    service = QiskitRuntimeService(channel="ibm_quantum", token="",
                                  instance=instance)
    # backend = service.least_busy(operational=True, simulator=False)
    backend = service.backend("ibm_fez") # fez
    device = backend.name
    backend_name = backend.name
    print(backend.name)


lattice = np.array([[ 1.233862, -2.137112,  0.      ],
                       [ 1.233862,  2.137112,  0.      ],
                       [ 0.      ,  0.      ,  8.685038]])
    
graphene = Structure(lattice, species=['C','C'], coords=[[2/3, 1/3, 0. ],[1/3, 2/3, 0.]])
graphene = SpacegroupAnalyzer(graphene).get_conventional_standard_structure()

n_supercell = n_supercell
scaling_matrix = np.identity(3)*n_supercell
scaling_matrix[2][2] = 1
graphene_supercell = copy.deepcopy(graphene)
graphene_supercell.make_supercell(scaling_matrix)
structure = graphene_supercell

# Defining lists accumulated results over repeats
big_df = pd.DataFrame()
combs = 0
nsuper_ls = []
vacs_ls = []
lambda_1_ls = []
ansatz_depth_ls = []
optimizer_ls = []
alpha_ls = []
runTime_ls = []
QPUtime_ls = []
valid_ls = []
num_hits_ls = []
num_hits_genuine_ls = []
FINAL_QOS_ls = []
FINAL_QOS_tot_ls = []
iters_ls = []
AR_ls = []
AR_exp_ls = []
avg_E_pre_ls = []
rnge_ls = []
avg_E_post_ls = []

obj_vals_accum = []
fvals_accum = []
probs_accum = []
fvals_post_sel_accum = []
probs_post_sel_accum = []


Q = build_qubo_vacancies(graphene_supercell, num_vac=num_vac, coord_obj=False, lambda_1 = lambda_1, beta=0)

for i in range(repeats):
        # Read in objective function values
        flipped_objectives = np.loadtxt(f'RawResults/costevals_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV3_l{lambda_1}_shts{shots}_{timestr}_rep{i}.txt') #
        obj_vals_accum.append(flipped_objectives)
        # Convert to numpy array for easier computation
        obj_vals_array = np.array(flipped_objectives)
        
        # Minimum QUBO energies which can be calculated from the modified version of brute force with a hard constraint
        if n_supercell == 3 and num_vac ==3:
            min_energy = -20
        elif n_supercell == 2 and num_vac ==3:
            min_energy = -5
        elif n_supercell == 4 and num_vac ==3:
            min_energy = -41
        elif n_supercell == 5 and num_vac ==3:
            min_energy = -68
        elif n_supercell == 6 and num_vac ==3:
            min_energy = -101
        elif n_supercell == 7 and num_vac ==3:
            min_energy = -140
        else:
            print('Classically find the correct mimimum energy!!')
            
        # Convergence plots
        plt.figure(figsize=(10, 5))
        plt.plot([0, maxiters], [min_energy, min_energy], "r--", linewidth=2, label="Min QUBO Energy") # 
        plt.plot(obj_vals_array,  linewidth=2, color='navy') #label="alpha = %.2f" % alpha,
        # Error bars every 5 iterations
        plt.legend(loc="upper right")
        plt.xlim(0, maxiters)  # Adjust xlim to match your data
        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")
        # plt.title(f"Repeat {i}")
        # plt.grid(True)
        plt.savefig(f'VQE_plots/loss_VQEcVAR{backend_name}_nsuper{n_supercell}_nV{num_vac}_rep{i}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        # Read in QPU time
        QPUtime = np.loadtxt(f'RawResults/QPUtime_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV3_l{lambda_1}_shts{shots}_{timestr}_rep{i}.txt')
        # Read in user runtime
        Usertime = np.loadtxt(f'RawResults/time_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV3_l{lambda_1}_shts{shots}_{timestr}_rep{i}.txt')
        # Read in time stamps to calculate queue time
        with open(f"RawResults/timeSTAMPS_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV3_l{lambda_1}_shts{shots}_{timestr}_rep{i}.txt", 'r') as file:
           dict_str = file.read()
        
        # Read in bitstrings
        bitstrings = np.loadtxt(f'RawResults/bitstrings_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV3_l{lambda_1}_shts{shots}_{timestr}_rep{i}.txt') #
        # Read in
        prob = np.loadtxt(f'RawResults/prob_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV3_l{lambda_1}_shts{shots}_{timestr}_rep{i}.txt') #
        
        # Calculate energies
        fvals = []
        for i in range(len(bitstrings)):
            # Convert bitstring to a 1D NumPy array
            x = np.array([int(bit) for bit in bitstrings[i]])
            xT = np.transpose(x)
            # Calculate energy = x^T * Q * x
            fvals.append(np.dot(xT, np.dot(Q, x)))
        
        # Create the DataFrame
        df_orig = pd.DataFrame({
            'x': bitstrings.tolist(),
            'fval': fvals,
            'probability': prob.tolist()
        })
        
        
        
        # Dataframe manipulation
        df = df_orig
        # Normalise probability
        df['probability'] = df['probability'].apply(lambda x: x/shots)
        df = df.rename(columns={"probability": "Validity % Occurence","fval": "energy"})
        df["energy"] = df["energy"]+((Q.shape[0]-num_vac)**2)*lambda_1
        pd.set_option('max_colwidth', 100)
        df = df.sort_values(by="Validity % Occurence", ascending=False)
        df['iters'] = len(flipped_objectives)
        cost_evals = len(flipped_objectives)
        
        min_E_exp = df['energy'].min()
        max_E_exp = df['energy'].max()
        
        rnge = np.sqrt((max_E_exp - min_E_exp)**2)
        
        # Calculate weighted energy
        df['Weighted Energy'] = df['energy'] * (df['Validity % Occurence'])
        # Calculate average energy
        avg_E = df['Weighted Energy'].sum()
        
        df['Validity % Occurence'] = df['Validity % Occurence']*100
        
        

        # Extract fval and probabilities
        fvals = df_orig['fval'].tolist()
        probabilities = df_orig['probability'].tolist()
        
        fvals_accum.append(fvals)
        probs_accum.append(probabilities)
        
        # Flatten lists if necessary
        fvals = np.asarray(fvals) + ((Q.shape[0]-num_vac)**2)*lambda_1 # Concatenate lists into a single array
        probs = np.asarray(probabilities)   # Concatenate lists into a single array
        
        # Accumulate probabilities for each unique fval
        accumulated_probs = {}
        for fval, prob in zip(fvals, probs):
            if fval in accumulated_probs:
                accumulated_probs[fval] += prob
            else:
                accumulated_probs[fval] = prob
        
        # Extract unique fvals and accumulated probabilities
        unique_fvals = np.array(list(accumulated_probs.keys()))
        unique_probs = np.array(list(accumulated_probs.values()))

        # Make final accumulated plot
        plt.figure(figsize=(12, 6))
        plt.bar(unique_fvals, unique_probs, color='navy')
        plt.axvline(x=min_energy, color='r', linestyle='--', label="Min QUBO Energy")
        plt.legend(loc="upper right")
        plt.xlabel('Energy')
        plt.ylabel('Probability')
        plt.title(f'Repeat {i}')
        plt.show()
        
        
        # Filter the DataFrame to include only rows with the minimum energy value
        good_df = df[df['energy'] == min_energy]
        num_hits = good_df['Validity % Occurence'].sum()
        
        start = timeit.default_timer()
        # Post-selection
        location_list = []
        for j in range(len(df)):
            if np.sum(df['x'][j]) == Q.shape[0]-num_vac:
                location_list.append(j)
            else:
                continue
            
        df = df = df.loc[location_list]
        
        # Stop timer
        stop = timeit.default_timer()
        time = stop - start
        print('Approx post selection time',time)
        
    
        valid = df['Validity % Occurence'].sum(axis=0)
        print('Validity is:',valid)
        
        # Calculate Ps (num_hits_genuine)
        good_df = df[df['energy'] == min_energy]
        num_hits_genuine = good_df['Validity % Occurence'].sum()
        
        # Now find Ps after post selection
        # Step 1: Calculate the total sum of the '% Occurrence' column
        total_sum = df['Validity % Occurence'].sum()
        # Step 2: Divide each value in the '% Occurrence' column by the total sum
        df['% Occurence'] = df['Validity % Occurence'] / total_sum
        # Step 3: Multiply each value by 100
        df['% Occurence'] = df['% Occurence'] * 100
        
        
        fvals_post_sel = np.asarray(df['energy'])
        probs_post_sel = np.asarray(df['Validity % Occurence'])/100
        
        fvals_post_sel_accum.append(fvals_post_sel)
        probs_post_sel_accum.append(probs_post_sel)
        
        # Calculate weighted energy
        df['Weighted Energy'] = df['energy'] * (df['% Occurence'] / 100)
        # Calculate average energy
        avg_E = df['Weighted Energy'].sum()
        print("Average Energy for SAR:", avg_E)
        
        avg_E_post = avg_E
        
        
        min_E_exp = df['energy'].min()
        max_E_exp = df['energy'].max()
        
        # Note max_E_exp will coincide with max_E from brute force (constrained)
        AR = (avg_E-max_E_exp)/(min_energy - max_E_exp)
        
        print('max_E_exp post-sel',max_E_exp)
        print('min_E_exp post-sel',min_E_exp)
        print('avg_E post-sel',avg_E)
        
        # Filter the DataFrame to include only rows with the minimum energy value
        good_df = df[df['energy'] == min_energy]
        QOS = good_df['% Occurence'].sum()
        print('QOS is:',QOS)
        
        
        big_df = pd.concat([big_df, df], ignore_index=True)
        # Define the path for saving the CSV file
        path = f'RawResults/VQEcVAR{backend_name}_nsuper{n_supercell}_nV{num_vac}_its{maxiters}_{device}_{ansatz_name}_p{p}_{obj_fun}alph{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_l{lambda_1}_rep{i}_{timestr}.csv'
        # Save the DataFrame to a CSV file
        big_df.to_csv(path, index=False)
        
        
        time_format = '%Y%m%d-%H%M%S'
        # Use ast.literal_eval to safely convert the string into a dictionary
        time_stamp_dict = ast.literal_eval(dict_str)
        
        # Convert the strings to datetime objects
        time0 = datetime.strptime(time_stamp_dict['timestr0thIT'], time_format)
        time1 = datetime.strptime(time_stamp_dict['timestr1stIT'], time_format)
        
        # Calculate the difference between the two times
        time_difference = time1 - time0
        
        # Convert time_difference to total seconds
        time_difference_seconds = time_difference.total_seconds()
        
        # Subtract the time difference in seconds from tot_time
        actual_runtime = Usertime - time_difference_seconds
        
         
        lambda_1_ls.append(lambda_1)
        # ansatz_depth_ls.append(ansatz_depth)
        alpha_ls.append(alpha)
        runTime_ls.append(actual_runtime) # usertime
        QPUtime_ls.append(QPUtime)
        vacs_ls.append(num_vac)
        nsuper_ls.append(n_supercell)
        num_hits_ls.append(num_hits)
        num_hits_genuine_ls.append(num_hits_genuine)
        valid_ls.append(valid)
        FINAL_QOS_ls.append(QOS)
        iters_ls.append(cost_evals)
        AR_ls.append(AR)
        rnge_ls.append(rnge)
        avg_E_post_ls.append(avg_E_post)
        
        combs = combs + 1
        
        
# Plot convergence
# First trim each list to the size of the smallest list, determine the length of the shortest list
min_length = min(len(lst) for lst in obj_vals_accum)
# Trim each list to the length of the shortest list
obj_vals_accum_trimmed = [lst[:min_length] for lst in obj_vals_accum]
# Convert to numpy array for easier computation
obj_vals_array = np.array(obj_vals_accum_trimmed)
# Compute average across the lists (mean)
average_list = np.mean(obj_vals_array, axis=0)
# Compute standard deviation across the lists
std_deviation_list = np.std(obj_vals_array, axis=0)

# Plot
plt.figure(figsize=(10, 5))
plt.plot([0, maxiters], [min_energy, min_energy], "r--", linewidth=2, label="Min QUBO Energy") # 
plt.plot(average_list,  linewidth=2, color='navy') #label="alpha = %.2f" % alpha,
# Error bars every 5 iterations
error_indices = np.arange(0, len(average_list), 5) 
plt.errorbar(error_indices, average_list[error_indices], yerr=std_deviation_list[error_indices],fmt='o',color='navy') #label="alpha = %.2f" % alpha, linewidth=2, fmt='o', capsize=5
plt.legend(loc="upper right")
plt.xlim(0, maxiters)  # Adjust xlim to match your data
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
# plt.title("History of Objective Values with Error Bars")
# plt.grid(True)
plt.savefig(f'VQE_plots/loss_VQEcVAR{backend_name}_nsuper{n_supercell}_nV{num_vac}_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_l{lambda_1}_{timestr}.pdf', format='pdf', bbox_inches='tight')
plt.show()


############ Plot distributions pre post selection
# Flatten lists if necessary
fvals_accum_flat = np.concatenate(fvals_accum) + ((Q.shape[0]-num_vac)**2)*lambda_1 # Concatenate lists into a single array
probs_accum_flat = np.concatenate(probs_accum) / repeats  # Concatenate lists into a single array

# Accumulate probabilities for each unique fval
accumulated_probs = {}
for fval, prob in zip(fvals_accum_flat, probs_accum_flat):
    if fval in accumulated_probs:
        accumulated_probs[fval] += prob
    else:
        accumulated_probs[fval] = prob

# Extract unique fvals and accumulated probabilities
unique_fvals = np.array(list(accumulated_probs.keys()))
unique_probs = np.array(list(accumulated_probs.values()))
# # Flatten lists if necessary

# Save distribution data
np.savetxt(f'distr_data/Energy_data_prepro_VQEcVAR{backend_name}_nsuper{n_supercell}_nV{num_vac}_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_l{lambda_1}_{timestr}.txt', unique_fvals, fmt='%.10f')
np.savetxt(f'distr_data/Probs_data_prepro_VQEcVAR{backend_name}_nsuper{n_supercell}_nV{num_vac}_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_l{lambda_1}_{timestr}.txt', unique_probs, fmt='%.10f')

# Make final accumulated plot of the pre post selected distribution
plt.figure(figsize=(12, 6))
plt.bar(unique_fvals, unique_probs, color='navy')
plt.axvline(x=min_energy, color='r', linestyle='--', label="Min QUBO Energy")
plt.legend(loc="upper right")
plt.xlabel('Energy')
plt.ylabel('Probability')
# plt.title('Probability vs. fval')
plt.xlim(-1+min_energy, 20)
plt.savefig(f'VQE_plots/distr_VQEcVAR{backend_name}_nsuper{n_supercell}_nV{num_vac}_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_l{lambda_1}_{timestr}.pdf', format='pdf', bbox_inches='tight')
plt.show()


############ Now post selected distributions
# Flatten lists if necessary
fvals_accum_flat = np.concatenate(fvals_post_sel_accum) # Concatenate lists into a single array
probs_accum_flat = np.concatenate(probs_post_sel_accum) / repeats  # Concatenate lists into a single array

# Accumulate probabilities for each unique fval
accumulated_probs = {}
for fval, prob in zip(fvals_accum_flat, probs_accum_flat):
    if fval in accumulated_probs:
        accumulated_probs[fval] += prob
    else:
        accumulated_probs[fval] = prob

# Extract unique fvals and accumulated probabilities
unique_fvals = np.array(list(accumulated_probs.keys()))
unique_probs = np.array(list(accumulated_probs.values()))

# Save distribution data
np.savetxt(f'distr_data/Energy_data_prepro_VQEcVAR{backend_name}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_{timestr}.txt', unique_fvals, fmt='%.10f')
np.savetxt(f'distr_data/Probs_data_prepro_VQEcVAR{backend_name}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_{timestr}.txt', unique_probs, fmt='%.10f')

# Make final accumulated plot
plt.figure(figsize=(12, 6))
plt.bar(unique_fvals, unique_probs*1/np.sum(unique_probs), color='navy')
plt.axvline(x=min_energy, color='r', linestyle='--', label="Min QUBO Energy")
plt.legend(loc="upper right")
plt.xlabel('Energy')
plt.ylabel('Probability')
# plt.title('Probability vs. fval')
plt.savefig(f'VQE_plots/distrPOStsel_VQEcVAR{backend_name}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_{timestr}.pdf', format='pdf', bbox_inches='tight')
plt.show()

    
# Creating an empty dictionary
myDict = {}
 
# Adding list as value
myDict['Num hits'] = num_hits_ls
myDict['Num hits genuine'] = num_hits_genuine_ls
myDict['QOS'] = FINAL_QOS_ls
myDict['Validity'] = valid_ls
myDict['Time'] = runTime_ls
myDict['QPU Time'] = QPUtime_ls
myDict['lambda'] = lambda_1_ls
# myDict['depth'] = ansatz_depth_ls
# myDict['optimizer'] = optimizer_ls
myDict['alpha'] = alpha_ls
myDict['nsuper'] = nsuper_ls
myDict['Vacancies'] = vacs_ls
myDict['iters'] = iters_ls
myDict['AR'] = AR_ls
myDict['avg_E_pre'] = avg_E_pre_ls
myDict['Range'] = rnge_ls
myDict['avg_E_post'] = avg_E_post_ls
# myDict['Median CNOT Error'] = median_cnot_error
# myDict['Median SX Error'] = median_sx_error
# myDict['Median Readout Error'] = median_readout_error

FINALdf = pd.DataFrame(myDict)

# Group and aggregate into lists
FINALdf = FINALdf.groupby(['lambda', 'alpha', 'nsuper', 'Vacancies'], as_index=False).agg({'Num hits': list,'Num hits genuine': list,'QOS': list,'Validity': list,\
                                                                                                                  'Time': list, 'QPU Time': list,'iters':list,'AR':list,'Range':list,'avg_E_post':list})

# Add columns
FINALdf['Num_hits_avg'] = FINALdf['Num hits'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['Num_hits_genuine_avg'] = FINALdf['Num hits genuine'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['QOS_avg'] = FINALdf['QOS'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['Validity_avg'] = FINALdf['Validity'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['Time_avg'] = FINALdf['Time'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['QPU_Time_avg'] = FINALdf['QPU Time'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['iters_avg'] = FINALdf['iters'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['AR_avg'] = FINALdf['AR'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['Range_avg'] = FINALdf['Range'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)
FINALdf['avg_E_post_avg'] = FINALdf['avg_E_post'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else None)

# Add columns
FINALdf['Num_hits_std'] = FINALdf['Num hits'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['Num_hits_genuine_std'] = FINALdf['Num hits genuine'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['QOS_std'] = FINALdf['QOS'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['Validity_std'] = FINALdf['Validity'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['Time_std'] = FINALdf['Time'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['QPU_Time_std'] = FINALdf['QPU Time'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['AR_std'] = FINALdf['AR'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['Range_std'] = FINALdf['Range'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)
FINALdf['avg_E_post_std'] = FINALdf['avg_E_post'].apply(lambda x: pd.Series(x).std() if len(x) > 1 else None)


# Calculate standard error
y = FINALdf['Num_hits_genuine_avg'].values/100
yerr = 2 * np.sqrt((y * (1-y)) / (shots*repeats+1))
FINALdf['yerr (not %)'] = yerr

y = FINALdf['QOS_avg'].values/100
yerr_postsel = 2 * np.sqrt((y * (1-y)) / (shots*repeats+1))
FINALdf['yerr post-sel (not %)'] = yerr_postsel

print(FINALdf)
FINALdf = FINALdf.sort_values(by="QOS_avg", ascending=False)
print('Best combination is',FINALdf.iloc[0])

# Define the path for saving the CSV file
path = f'final_CSVs/VQE{backend_name}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_{lambda_1}_{timestr}.csv'
# Save the DataFrame to a CSV file
FINALdf.to_csv(path, index=False, float_format='%.15g')
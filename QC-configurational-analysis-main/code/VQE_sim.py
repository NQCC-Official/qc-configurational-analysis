# -*- coding: utf-8 -*-
"""

Solving the graphene defect problem using VQE on a simulator

"""

# General imports
import numpy as np

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2, RealAmplitudes

# SciPy minimizer routine
from scipy.optimize import minimize
# Plotting functions
import matplotlib.pyplot as plt

from qiskit.circuit.library import QAOAAnsatz
# from qiskit.opflow import I, X, Y, PauliSumOp
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from itertools import combinations

# runtime imports
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
# from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_aer import AerSimulator
import qiskit_aer

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.vis import structure_vtk
import copy
from quantum_computing_functions import *
from quantum_computing_postprocessing import *

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_ising, to_ising

from qiskit_algorithms.optimizers import SPSA

import timeit

import qiskit_ibm_runtime as qiskit_ibm_runtime
import qiskit as qiskit

from scipy.optimize import OptimizeResult

import cProfile
import io
import pstats
import sys

print(qiskit_ibm_runtime.__version__)
print(qiskit.__version__)
print(qiskit_aer.__version__)

sim = 'Y' # Y N Y_noisy 
# Y_noisy can be used as a noise model and to test before using runtime (On feasible to run for n_supercell =2,3. 4 takes a long time).

# Set problem size
n_supercell = 3
num_vac = 3

# Hyperparameters
lambda_1 = 3
shots = 10000 
maxiters = 10
tol = 1 # tolerance of COBYLA (convergence criteria)
optimization_level = 3 # compilation effort

optimiser_name = 'cobyla'
obj_fun = 'CVAR' #'EV' 'CVAR', 'ACVAR'
ansatz_name = 'RealAmp' # QAOA HEA RealAmp QAOAmixer
p = 2

alpha = 0.4 # CVaR parameter
# aCVaR parameters
acvar_min = 0.2
acvar_max = 0.7

sim_method = "statevector" #'density_matrix',"statevector","matrix_product_state",extended_stabilizer
# Use MPS for 32 variable (n_super=4)

repeats = 1


if sim == 'Y':
    sim_method = sim_method
    aer_sim = AerSimulator(method=sim_method)
    backend = aer_sim
    device = 'StV'
    
elif sim == 'Y_noisy':
    # Get noise model from real device
    instance = ''
    service = QiskitRuntimeService(channel="ibm_quantum", token="",
                                  instance=instance) 
    real_backend = service.backend("ibm_fez")
    noise_model = NoiseModel.from_backend(real_backend)
    backend = AerSimulator(method='matrix_product_state',
                        noise_model=noise_model)
    
    device = 'Fake '+real_backend.name
    
    print(device)

# Can run on QPU here but VQE_QPU.py and serverless.py were used
else:
    instance = ''
    # To run on hardware, select the backend with the fewest number of jobs in the queue
    service = QiskitRuntimeService(channel="ibm_quantum", token="",
                                  instance=instance) # 
    # backend = service.least_busy(operational=True, simulator=False)
    backend = service.backend("ibm_fez")
    device = backend.name
    print(backend.name)

# Need to change the way acvar parameters are updated with SPSA as it has double the number of iterations of other optimisers
if optimiser_name == 'SPSA':
    acvar_range = acvar_max - acvar_min
    acvar_increment = acvar_range/(maxiters*2)
else:
    acvar_range = acvar_max - acvar_min
    acvar_increment = acvar_range/maxiters
 
# Define graphene cell - taken from Bruno Camino's code
lattice = np.array([[ 1.233862, -2.137112,  0.      ],
                        [ 1.233862,  2.137112,  0.      ],
                        [ 0.      ,  0.      ,  8.685038]])
    
graphene = Structure(lattice, species=['C','C'], coords=[[2/3, 1/3, 0. ],[1/3, 2/3, 0.]])
graphene = SpacegroupAnalyzer(graphene).get_conventional_standard_structure()

scaling_matrix = np.identity(3)*n_supercell
scaling_matrix[2][2] = 1
graphene_supercell = copy.deepcopy(graphene)
graphene_supercell.make_supercell(scaling_matrix)
structure = graphene_supercell


if obj_fun == 'CVAR':
    alpha = alpha
elif obj_fun == 'ACVAR':
    alpha_acvar = acvar_min


def xy_mixer(num_qubits):
    """
    Create an XY mixer for QAOA with num_qubits.
    The XY mixer will apply the X and Y Pauli operators to each qubit.

    Parameters:
        num_qubits (int): Number of qubits

    Returns:
        SparsePauliOp: The XY mixer operator.
    """
    xy_terms = []
    
    for qubit in range(num_qubits):
        # Constructing the Pauli operators in terms of strings
        x_term = 'X' + 'I' * (num_qubits - 1 - qubit) + 'I' * qubit
        y_term = 'Y' + 'I' * (num_qubits - 1 - qubit) + 'I' * qubit
        
        # Pauli terms with qubit indices and coefficients
        xy_terms.append((x_term, [qubit], 1.0))  # X operator on the current qubit
        xy_terms.append((y_term, [qubit], 1.0))  # Y operator on the current qubit
    
    # Use SparsePauliOp.from_sparse_list to create the Pauli operator for the XY mixer
    xy_mixer_op = SparsePauliOp.from_sparse_list(xy_terms, num_qubits=num_qubits)
    
    return xy_mixer_op


    
Q = build_qubo_vacancies(graphene_supercell, num_vac=num_vac, coord_obj=False, lambda_1 = lambda_1, beta=0)
# print('Lambda is: ',lambda_1)
# print(Q)
# Save Q matrix (can use load in this saved file in VQE_QPU.py)
np.savetxt(f"Q_n{n_supercell}_l{lambda_1}.txt", Q, delimiter=",")

qp = QuadraticProgram()
# Create binary variables 
[qp.binary_var() for _ in range(Q.shape[0])]
# Spec problem type
qp.minimize(quadratic=Q)

qp_orig = qp

_, offset = qp.to_ising()
print('Offset:',offset)

qubit_op = to_ising(QuadraticProgramToQubo().convert(qp))[0]
hamiltonian = qubit_op

print('Hamiltonian:',hamiltonian)
# Save to a text file
np.savetxt(f'hamiltonian_n{n_supercell}_l{lambda_1}.txt', hamiltonian.coeffs, fmt='%d')

# Define ansatz
if ansatz_name == 'RealAmp':
    ansatz = RealAmplitudes(qubit_op.num_qubits,reps=1)
elif ansatz_name == 'HEA':
    ansatz = EfficientSU2(qubit_op.num_qubits,reps=1) 
elif ansatz_name == 'QAOA':
    tol = 1e-7 # QAOA requires a smaller tolerance
    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=p) 
elif ansatz_name == 'QAOAmixer':
    # Create the XY mixer
    xy_mixer = xy_mixer(Q.shape[0])
    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=p,mixer_operator = xy_mixer) 

# Add measurements (Computational pauli Z basis)
ansatz.measure_all()
# print('Ansatz depth is:',ansatz.decompose().decompose().depth())
print('Ansatz depth is:',ansatz.decompose().decompose().depth()) #.decompose().decompose()
# print(ansatz.decompose().decompose())
ansatz.decompose().decompose().draw("mpl", style="iqp", filename=f"ansatz_diagram_{ansatz_name}.png")

num_params = ansatz.num_parameters
print('Params:',num_params)

# Compile
pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
ansatz = pm.run(ansatz)


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
        
    """
    start_time = time.time()  # Start timing
    
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    # !!! Can play about here with CVaR and aCVaR
    energy = result[0].data.evs[0]
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    cost_history_dict["cost_fun_time"].append(elapsed_time)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}], time taken: {elapsed_time}")

    return energy


def cost_func_cvar(params, ansatz, hamiltonian, estimator,alpha):
    """
    Function which computes the CVaR objective function.
    """
    
    start_time = time.time()  # Start timing
    pub = (ansatz,  [params]) 
    # Run ansatz at specified params
    result = sampler.run(pubs=[pub]).result()
    counts = result[0].data.meas.get_counts()
    
    bitstrings, prob = zip(*counts.items())
    # Convert binary bitstrings to arrays of 0s and 1s (base 2 representation)
    bitstrings = np.array([[int(bit) for bit in bitstring] for bitstring in bitstrings])
    
    # Run ansatz at specified params
    fvals = []
    for i in range(len(bitstrings)):
        # Convert bitstring to a 1D NumPy array
        x = np.array([int(bit) for bit in bitstrings[i]])
        xT = np.transpose(x)
        # Calculate energy = x^T * Q * x
        fvals.append(np.dot(xT, np.dot(Q, x))-offset)
    
    # prob = prob/shots
    prob = tuple(item / shots for item in prob)
    
    # Combine the two lists, sort by fval, and unpack back into two lists
    sorted_pairs = sorted(zip(fvals, prob), key=lambda x: x[0])
    fval_sorted, prob_sorted = zip(*sorted_pairs)
    
    # Convert back to lists (optional)
    fvals = list(fval_sorted)
    prob = list(prob_sorted)
    
    accum_percent = 0  # once alpha is reached, stop
    cvar = 0
    
    # Now take alpha fraction of bitstrings with lowest energy
    for i in range(len(prob)):          # prob of a pure state and the objective value of that state
        cvar += fvals[i] * min(prob[i], alpha - accum_percent)
        accum_percent += prob[i]
        #print('EVCVar, P, E, EVCVAR',probability, Energyvalues,cvar)
       
        if accum_percent >= alpha :
            break
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    
    # Save to dictionaries
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(cvar/alpha)
    cost_history_dict["cost_fun_time"].append(elapsed_time)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {cvar/alpha}], time taken: {elapsed_time}")
    # Return CVaR objective value
    return cvar/alpha

def cost_func_acvar(params, ansatz, hamiltonian, estimator,acvar_min):
    """
    Function which computes the aCVaR objective function.
    """
    
    global alpha_acvar
    
    start_time = time.time()  # Start timing
    
    # Get results at specific ansatz and params
    pub = (ansatz,  [params]) #[hamiltonian],
    result = sampler.run(pubs=[pub]).result()
    counts = result[0].data.meas.get_counts()
    
    bitstrings, prob = zip(*counts.items())
    # Convert binary bitstrings to arrays of 0s and 1s (base 2 representation)
    bitstrings = np.array([[int(bit) for bit in bitstring] for bitstring in bitstrings])
    
    # Get energies
    fvals = []
    for i in range(len(bitstrings)):
        # Convert bitstring to a 1D NumPy array
        x = np.array([int(bit) for bit in bitstrings[i]])
        xT = np.transpose(x)
        # Calculate energy = x^T * Q * x
        fvals.append(np.dot(xT, np.dot(Q, x))-offset)
    
    # prob = prob/shots
    prob = tuple(item / shots for item in prob)
    
    # Combine the two lists, sort by fval, and unpack back into two lists
    sorted_pairs = sorted(zip(fvals, prob), key=lambda x: x[0])
    fval_sorted, prob_sorted = zip(*sorted_pairs)
    
    # Convert back to lists (optional)
    fvals = list(fval_sorted)
    prob = list(prob_sorted)
    
    accum_percent = 0  # once alpha is reached, stop
    cvar = 0
    
    # Get aCVaR fraction of bitstring energies
    for i in range(len(prob)):          # prob of a pure state and the objective value of that state
        cvar += fvals[i] * min(prob[i], alpha_acvar - accum_percent)
        accum_percent += prob[i]
        #print('EVCVar, P, E, EVCVAR',probability, Energyvalues,cvar)
       
        if accum_percent >= alpha_acvar :
            break
    
    alpha_acvar += acvar_increment
    
    print('Next alpha:',alpha_acvar)
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(cvar/alpha_acvar
)
    cost_history_dict["cost_fun_time"].append(elapsed_time)
    
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {cvar/alpha_acvar}], time taken: {elapsed_time}")
    
    return cvar/alpha_acvar


# Global dictionary to store timing and iteration data
timing_info = {"iteration_times": [], "start_time": None}

def iteration_callback(xk):
    """
    Callback function to record the time taken for each iteration.
    """
    if timing_info["start_time"] is not None:
        elapsed_time = time.time() - timing_info["start_time"]
        timing_info["iteration_times"].append(elapsed_time)
        print(f"Iteration completed in {elapsed_time:.6f} seconds")
    # Reset the timer for the next iteration
    timing_info["start_time"] = time.time()

# Example gradient estimation function for SPSA
def grad(LossFunction, w, c, args):
    """Estimate the gradient using simultaneous perturbation."""
    n = len(w)
    delta = np.random.choice([-1, 1], size=n)  # Random perturbation
    # Perturb the parameters
    w_plus = w + c * delta
    w_minus = w - c * delta
    # Approximate the gradient
    grad_estimate = (LossFunction(w_plus, *args) - LossFunction(w_minus, *args)) / (2 * c) * delta
    return grad_estimate

# Example function to initialize hyperparameters (you can adjust as needed)
def initialize_hyperparameters(alpha, LossFunction, w, N_iterations):
    # Example initialization of hyperparameters
    a = 0.1  # Starting value for ak
    A = 10  # Scaling factor for the learning rate
    c = 0.1  # Starting value for ck
    return a, A, c

# Custom SPSA optimizer
def SPSA(LossFunction, x0, args=(), alpha=0.602, gamma=0.101, N_iterations=maxiters, **kwargs):
    w = np.array(x0, dtype=float)  # Initial parameters as NumPy array
    a, A, c = initialize_hyperparameters(alpha, LossFunction, w, N_iterations)
    
    for k in range(1, N_iterations + 1):
        # Update ak and ck
        ak = a / ((k + A)**(alpha))
        ck = c / (k**(gamma))
        
        # Estimate the gradient
        gk = grad(LossFunction, w, ck, args)  # Pass the additional arguments here
        
        # Update the parameters using the gradient
        w -= ak * gk
        
        # Convergence check (optional)
        if np.linalg.norm(gk) < kwargs.get('tol', 1e-6):
            break

    # Prepare the result to match the structure of scipy.optimize result
    result = OptimizeResult()
    result.x = w
    result.fun = LossFunction(w, *args)  # Make sure to pass arguments here as well
    result.success = True
    result.message = 'Optimization converged' if np.linalg.norm(gk) < kwargs.get('tol', 1e-15) else 'Maximum iterations reached'
    return result


# Initialize cost_history_dict at the start of your script or function
cost_history_dict = {
    "iters": 0,
    "prev_vector": None,
    "cost_history": [],
    "cost_fun_time": []  # Initialize cost_fun_time as an empty list
}

# Set random starting params
x0 = 2 * np.pi * np.random.random(num_params)
print(x0)

# Selecting optimiser
if optimiser_name=='cobyla':
    opt='cobyla'
else:
    opt= SPSA
    
# if sim = Y, then backend is AerSimulator(method=sim_method), and an actual session doesn't open
with Session(backend=backend) as session:
    for i in range(repeats):
        
        startBIGGEST = timeit.default_timer()
        
        print(f'Repeat {i}')
        estimator = Estimator(mode=session) #
        estimator.options.default_shots = shots
        
        sampler = Sampler(mode=session) #,options={'shots': shots}
        sampler.options.default_shots = shots
        # estimator.options.default_shots = 10000
    
        # Create a StringIO object to capture the output
        output = io.StringIO()
        
        # Redirect the standard output to the StringIO object
        sys.stdout = output
        
        # Create a profiler instance
        profiler = cProfile.Profile()
        
        # Ensure any active profiler is disabled before starting a new one
        if profiler.getstats():
            profiler.disable()
        
        # profiler.disable()
        # Start profiling
        profiler.enable()
        
        # Minimisation routine dependent on objective function being used
        if obj_fun == 'CVAR':
            res = minimize(
                cost_func_cvar,
                x0,
                callback=iteration_callback,
                args=(ansatz, hamiltonian, sampler,alpha),
                method=opt,options={'maxiter': maxiters,'tol':tol,'disp': True,'verbose': 1} # set a tolerance instead!!
            ) # ,'catol':catol
            
       
        elif obj_fun == 'ACVAR':
            res = minimize(
                cost_func_acvar,
                x0,
                args=(ansatz, hamiltonian,sampler,acvar_min),
                method=opt,options={'maxiter': maxiters,'tol':tol,'disp': True} # set a tolerance instead!!
            ) 
        
        else:
            res = minimize(
                cost_func,
                x0,
                args=(ansatz, hamiltonian, estimator),
                method=opt,options={'maxiter': maxiters,'tol':tol,'disp': True,'verbose': 1} # set a tolerance instead!!
            )
        
        # Stop profiling
        profiler.disable()

        # Print the profiling results
        stats = pstats.Stats(profiler)
        stats.strip_dirs()  # Removes extraneous path info
        stats.sort_stats('cumulative')  # Sort by cumulative time (or 'time' for per-call time)
        stats.print_stats(40)  # Print the top 20 lines
        # Restore the standard output back to the console
        sys.stdout = sys.__stdout__
        # Convert the output to a string
        profiling_output = output.getvalue()
        
        # Now you can store it in a dictionary
        profiling_dict = {
            'profiling_output': profiling_output
        }
        
        # Example: print the dictionary
        print(profiling_dict)
        print(res)
        print('Rescaled energy is:',res.fun + ((Q.shape[0]-num_vac)**2)*lambda_1+offset)
        print('Final parameters:',res.x)
        
        # Assign solution parameters to ansatz
        qc = ansatz.assign_parameters(res.x)
        # Add measurements to our circuit
        qc.measure_all()
    
        
        # Sample ansatz at optimal parameters
        samp_dist = sampler.run([qc]).result() #.quasi_dists[0]
        samp_dist = samp_dist[0].data.meas.get_counts()
    

    
        # Plot convergence
        fig, ax = plt.subplots()
        adjustment = ((Q.shape[0] - num_vac) ** 2) * lambda_1 + offset
        adjusted_costs = np.array(cost_history_dict["cost_history"]) + adjustment
        ax.plot(range(cost_history_dict["iters"]), adjusted_costs)
        ax.axhline(y=-20, color='red', linestyle='--')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        plt.draw()
        
        # Save final params
        path = f'RawResults/params_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt'
        with open(path, 'w') as file:
            file.write(str(res.x))
            
            
        samp_dist = dict(sorted(samp_dist.items(), key=lambda item: item[1], reverse=True))
        
        np.savetxt(f'RawResults/costevals_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt', adjusted_costs , fmt='%.10f')
                    
        # Separate the sorted data into two lists: bitstrings and probabilities
        # Unzip the dictionary into two lists (bitstrings and probabilities)
        bitstrings, prob = zip(*samp_dist.items())
        # Convert binary bitstrings to arrays of 0s and 1s (base 2 representation)
        bitstrings = np.array([[int(bit) for bit in bitstring] for bitstring in bitstrings])
        
        
        np.savetxt(f'RawResults/bitstrings_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt', bitstrings)
        # np.savetxt(f'1.RawResults/fvals_VQEcVAR{backend_name}_nsuper{n_supercell}_nV{num_vac}_its{maxiter}_alph{alpha}_rep{j}.txt', fvals)
        np.savetxt(f'RawResults/prob_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt', prob)
        
        # Stop the overall timer and print the total runtime
        stopBIGGEST = timeit.default_timer()
        UserRuntime = stopBIGGEST - startBIGGEST
        print('User runtime', UserRuntime)
        
        
        # Time 
        path = f'RawResults/time_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt'
        with open(path, 'w') as file:
            file.write(str(UserRuntime))
            
        # Save cost function time
        np.savetxt(f'RawResults/costfunTime_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt', cost_history_dict["cost_fun_time"])
        # Save iteration times
        np.savetxt(f'RawResults/iterationTime_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt', timing_info["iteration_times"])
        # Save profiler output
        path = f'RawResults/profileTime_VQE_{device}_{ansatz_name}_p{p}_{obj_fun}_{alpha}_ac{acvar_min}-{acvar_max}_{optimiser_name}_tol{tol}_nsuper{n_supercell}_nV{num_vac}_l{lambda_1}_rep{i}.txt'
        # Open file in write mode
        with open(path, "w") as f:
            # Create stats object
            stats = pstats.Stats(profiler, stream=f)  # Pass the file stream to the Stats constructor
            stats.strip_dirs()  # Removes extraneous path info
            stats.sort_stats('cumulative')  # Sort by cumulative time (or 'time' for per-call time)
            stats.print_stats(40) 
        
        # Reset cost function parameters
        alpha_acvar = acvar_min
        cost_history_dict["iters"] = 0
        cost_history_dict["prev_vector"] = 0
        cost_history_dict["cost_history"] = []
        cost_history_dict["cost_fun_time"] = []
    

import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize

# Function to generate Bell state circuits with variable N (number of qubits)
def create_bell_circuit(N):
    qc = QuantumCircuit(N, N)  # N qubits and N classical bits
    qc.h(0)  # Apply Hadamard gate to the first qubit
    for i in range(N-1):
        qc.cx(i, i+1)  # Apply CNOT gates to entangle
    qc.measure(range(N), range(N))  # Add measurement gates
    return qc

# Define POVM elements (same as before, we can use the same for any N)
def povm_tetrahedron():
    E0 = np.array([[1, 0], [0, 0]]) / 2  # POVM element 1
    E1 = np.array([[0, 0], [0, 1]]) / 2  # POVM element 2
    E2 = np.array([[0.5, 0.5], [0.5, 0.5]]) / 2  # POVM element 3
    E3 = np.array([[0.5, -0.5], [-0.5, 0.5]]) / 2  # POVM element 4
    return [E0, E1, E2, E3]
# Simulate POVM for a given circuit and N qubits
def simulate_povm(qc, povm_elements, simulator, shots):
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts()
    
    # Initialize probabilities for each POVM element
    probabilities = []
    total_shots = sum(counts.values())
    
    # Define the states that correspond to measurement outcomes for N qubits
    measurement_states = {
        '0'*qc.num_qubits: np.kron(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])),
        '1'*qc.num_qubits: np.kron(np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 1]]))
    }
    
    # Iterate over each POVM element pair and calculate the probabilities
    for element1 in povm_elements:
        for element2 in povm_elements:
            prob = 0
            for key, state in measurement_states.items():
                prob += counts.get(key, 0) * np.real(np.trace(np.kron(element1, element2) @ state))
            probabilities.append(prob / total_shots)
    
    return probabilities

# Simulate POVM for a given circuit
#def simulate_povm(qc, povm_elements, simulator, shots):
#    result = execute(qc, simulator, shots=shots).result()
#    counts = result.get_counts()
#    probabilities = []
#    total_shots = sum(counts.values())
#    for element1 in povm_elements:
#        for element2 in povm_elements:
#            prob = counts.get('00', 0) * np.trace(np.kron(element1, element2) @ np.array([[1, 0], [0, 0]])) \
#                 + counts.get('11', 0) * np.trace(np.kron(element1, element2) @ np.array([[0, 0], [0, 1]]))
#            probabilities.append(prob / total_shots)
#    return probabilities

# Enforce positive semidefinite matrix
def enforce_physical_density_matrix(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)  # Set negative eigenvalues to 0
    rho = (eigvecs @ np.diag(eigvals) @ eigvecs.T.conj())
    return rho / np.trace(rho)  # Normalize trace to 1

# Likelihood function for MLE
def likelihood(rho, povm_elements, probabilities):
    rho = rho.reshape((4, 4))
    rho = (rho + rho.T.conj()) / 2  # Make Hermitian
    rho = enforce_physical_density_matrix(rho)  # Ensure physical state
    
    likelihood_value = 0
    for p, (E1, E2) in zip(probabilities, [(E1, E2) for E1 in povm_elements for E2 in povm_elements]):
        prob = np.real(np.trace(np.kron(E1, E2) @ rho))
        if prob > 0:
            likelihood_value += p * np.log(prob)
    return -likelihood_value  # Minimize negative likelihood

# MLE Optimization
def mle_optimization(probabilities, povm_elements, initial_guess):
    result = minimize(likelihood, initial_guess.flatten(), args=(povm_elements, probabilities), method='Powell')
    rho_mle = result.x.reshape((4, 4))
    rho_mle = enforce_physical_density_matrix(rho_mle)
    return rho_mle

# Run the comparison between Linear Inversion and MLE for different N
N_values = list(range(2, 6))  # Example N values (number of qubits)
fidelity_li_list = []
fidelity_mle_list = []
runtime_li_list = []
runtime_mle_list = []

# Ideal Bell state for comparison (for two qubits)
ideal_bell_state = np.array([[0.5, 0, 0, 0.5],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0.5, 0, 0, 0.5]])

# Noise model
noise_model = NoiseModel()
p_error_1q = 0.5
p_error_2q = 0.2
error_gate1 = depolarizing_error(p_error_1q, 1)
error_gate2 = depolarizing_error(p_error_2q, 2)
noise_model.add_all_qubit_quantum_error(error_gate1, ['x', 'h'])
noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])

# Simulator
simulator = Aer.get_backend('aer_simulator')
shots = 1024

# Run the simulation for each N
for N in N_values:
    qc = create_bell_circuit(N)
    
    # Simulate POVM measurements
    povm_elements = povm_tetrahedron()
    probabilities = simulate_povm(qc, povm_elements, simulator, shots)

    # Linear Inversion
    start_time_li = time.perf_counter()
    rho_li = sum(p * np.kron(E1, E2) for p, (E1, E2) in zip(probabilities, [(E1, E2) for E1 in povm_elements for E2 in povm_elements]))
    rho_li = (rho_li + rho_li.T.conj()) / 2  # Make Hermitian
    rho_li /= np.trace(rho_li)  # Normalize trace to 1
    end_time_li = time.perf_counter()
    runtime_li = end_time_li - start_time_li
    
    # MLE
    start_time_mle = time.perf_counter()
    rho_mle = mle_optimization(probabilities, povm_elements, rho_li)
    end_time_mle = time.perf_counter()
    runtime_mle = end_time_mle - start_time_mle
    
    # Calculate fidelities
    rho_li_dm = DensityMatrix(rho_li)
    rho_mle_dm = DensityMatrix(rho_mle)
    ideal_bell_state_dm = DensityMatrix(ideal_bell_state)
    
    fidelity_li = state_fidelity(rho_li_dm, ideal_bell_state_dm)
    fidelity_mle = state_fidelity(rho_mle_dm, ideal_bell_state_dm)
    
    # Store results
    fidelity_li_list.append(fidelity_li)
    fidelity_mle_list.append(fidelity_mle)
    runtime_li_list.append(runtime_li)
    runtime_mle_list.append(runtime_mle)

# Plotting fidelity and runtime for different N
fig, (ax1) = plt.subplots()
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Plot fidelity
ax1.plot(N_values, fidelity_li_list, label='Linear Regression', marker='o', linestyle='--', color='blue')
ax1.plot(N_values, fidelity_mle_list, label='MLE', marker='s', linestyle='--', color='orange')
ax1.set_ylabel('Fidelity')
ax1.set_xlabel('N (number of qubits)')
ax1.set_title('Fidelity Comparison for Linear Regression and MLE')
ax1.legend()
ax1.grid(True)

# Plot runtime
#ax2.plot(N_values, runtime_li_list, label='Linear Regression', marker='o', linestyle='--', color='blue')
#ax2.plot(N_values, runtime_mle_list, label='MLE', marker='s', linestyle='--', color='orange')
#ax2.set_ylabel('Runtime (s)')
#ax2.set_xlabel('N (number of qubits)')
#ax2.set_title('Runtime Comparison for Linear Regression and MLE')
#ax2.legend()
#ax2.grid(True)

# Show plots
plt.tight_layout()
plt.show()

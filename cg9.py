# Import necessary modules from Qiskit, SciPy, and NumPy
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import time

# Define the POVM tetrahedron elements
def povm_tetrahedron():
    E0 = np.array([[1, 0], [0, 0]]) / 2  # POVM element 1
    E1 = np.array([[0, 0], [0, 1]]) / 2  # POVM element 2
    E2 = np.array([[0.5, 0.5], [0.5, 0.5]]) / 2  # POVM element 3
    E3 = np.array([[0.5, -0.5], [-0.5, 0.5]]) / 2  # POVM element 4
    return [E0, E1, E2, E3]

# Function to generate Bell state circuits with error correction (three-qubit repetition code)
def create_bell_circuit_with_error_correction(N):
    qc = QuantumCircuit(3 * N, 3 * N)  # 3 physical qubits per logical qubit, and 3*N classical bits for measurement

    # Encode each logical qubit into three physical qubits (repetition code)
    for i in range(N):
        qc.h(3*i)  # Apply Hadamard gate to the first logical qubit (1st of 3 physical qubits)
        qc.cx(3*i, 3*i+1)  # Apply CNOT for encoding
        qc.cx(3*i, 3*i+2)  # Apply CNOT for encoding
    
    # Entangle the qubits using CNOT gates as before, but between logical qubits (first of each set of 3 qubits)
    for i in range(N-1):
        qc.cx(3*i, 3*(i+1))  # Entangle logical qubits

    # Add measurement gates
    for i in range(N):
        # Measure the three physical qubits for each logical qubit into separate classical bits
        qc.measure([3*i, 3*i+1, 3*i+2], [3*i, 3*i+1, 3*i+2])

    return qc

# Define a noise model for simulation
noise_model = NoiseModel()
p_error_1q = 0.5  # Example error rate for 1-qubit gates
p_error_2q = 0.2  # Example error rate for 2-qubit gates
error_gate1 = depolarizing_error(p_error_1q, 1)
error_gate2 = depolarizing_error(p_error_2q, 2)
noise_model.add_all_qubit_quantum_error(error_gate1, ['x', 'h'])
noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])
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

# Function to check if a matrix is a valid density matrix
def is_valid_density_matrix(rho):
    trace_rho = np.trace(rho)
    if not np.allclose(trace_rho, 1, atol=1e-6):
        print(f"Invalid trace: {trace_rho}")
        return False

    if not np.allclose(rho, rho.T.conj(), atol=1e-6):
        print("Matrix is not Hermitian")
        return False

    # Check if eigenvalues are non-negative
    eigvals = np.linalg.eigvalsh(rho)
    if np.any(eigvals < -1e-10):  # Tolerance for small negative eigenvalues due to numerical errors
        print(f"Invalid eigenvalues: {eigvals}")
        return False
    
    return True

# Define N values and run the circuit
N_values = list(range(2, 6))  # Example N values (number of qubits)
shots = 1024  # Number of shots for the simulation
simulator = Aer.get_backend('aer_simulator')  # Quantum simulator

fidelity_li_list = []
fidelity_mle_list = []
runtime_li_list = []
runtime_mle_list = []

# Ideal Bell state for comparison (for two qubits)
ideal_bell_state = np.array([[0.5, 0, 0, 0.5],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0.5, 0, 0, 0.5]])

for N in N_values:
    qc = create_bell_circuit_with_error_correction(N)
    povm_elements = povm_tetrahedron()
    probabilities = simulate_povm(qc, povm_elements, simulator, shots)

    # Linear Inversion
    start_time_li = time.perf_counter()
    rho_li = sum(p * np.kron(E1, E2) for p, (E1, E2) in zip(probabilities, [(E1, E2) for E1 in povm_elements for E2 in povm_elements]))
    rho_li = (rho_li + rho_li.T.conj()) / 2  # Make Hermitian
    trace_rho_li = np.trace(rho_li)
    if trace_rho_li > 1e-12:  # Check if trace is valid
        rho_li /= trace_rho_li  # Normalize trace to 1
    else:
        print("Trace of rho_li is too small, skipping normalization.")
        rho_li = np.eye(rho_li.shape[0])  # If trace is too small, use identity matrix
    end_time_li = time.perf_counter()
    runtime_li = end_time_li - start_time_li

    # MLE
    start_time_mle = time.perf_counter()
    rho_mle = mle_optimization(probabilities, povm_elements, rho_li)
    end_time_mle = time.perf_counter()
    runtime_mle = end_time_mle - start_time_mle

    # Check if density matrices are valid before calculating fidelity
    if is_valid_density_matrix(rho_li):
        rho_li_dm = DensityMatrix(rho_li)
        fidelity_li = state_fidelity(rho_li_dm, DensityMatrix(ideal_bell_state))
    else:
        print("rho_li is not a valid density matrix, skipping fidelity calculation for Linear Inversion.")
        fidelity_li = None

    if is_valid_density_matrix(rho_mle):
        rho_mle_dm = DensityMatrix(rho_mle)
        fidelity_mle = state_fidelity(rho_mle_dm, DensityMatrix(ideal_bell_state))
    else:
        print("rho_mle is not a valid density matrix, skipping fidelity calculation for MLE.")
        fidelity_mle = None

    # Store results if fidelity was calculated
    if fidelity_li is not None:
        fidelity_li_list.append(fidelity_li)
    if fidelity_mle is not None:
        fidelity_mle_list.append(fidelity_mle)
    
    runtime_li_list.append(runtime_li)
    runtime_mle_list.append(runtime_mle)

# Plotting fidelity and runtime for different N
fig, (ax1) = plt.subplots()
ax1.plot(N_values, fidelity_li_list, label='Linear Regression (With Error Correction)', marker='o', linestyle='--', color='blue')
ax1.plot(N_values, fidelity_mle_list, label='MLE (With Error Correction)', marker='s', linestyle='--', color='orange')
ax1.set_ylabel('Fidelity')
ax1.set_xlabel('N (number of qubits)')
ax1.set_title('Fidelity Comparison with Depolarizing Error Correction')
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.show()

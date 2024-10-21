import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize

# Step 1: Create a quantum circuit for Alice and Bob (2 qubits + 2 classical bits)
qc = QuantumCircuit(2, 2)  # 2 qubits and 2 classical bits

# Alice prepares a Bell state (entangled state |Φ+⟩ = (|00⟩ + |11⟩)/√2)
qc.h(0)  # Apply Hadamard gate to Alice's qubit
qc.cx(0, 1)  # Apply CNOT gate to entangle Alice's and Bob's qubits
qc.measure([0, 1], [0, 1])  # Add measurement gates

# Define POVM elements for a two-qubit system
def povm_tetrahedron():
    E0 = np.array([[1, 0], [0, 0]]) / 2  # POVM element 1
    E1 = np.array([[0, 0], [0, 1]]) / 2  # POVM element 2
    E2 = np.array([[0.5, 0.5], [0.5, 0.5]]) / 2  # POVM element 3
    E3 = np.array([[0.5, -0.5], [-0.5, 0.5]]) / 2  # POVM element 4
    return [E0, E1, E2, E3]

# Define a function to simulate POVM measurements
def simulate_povm(qc, povm_elements, simulator, shots):
    # Execute the quantum circuit and measure in the computational basis
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts()  # Get measurement counts
    
    # Calculate probabilities for each POVM element based on counts for two-qubit POVM
    probabilities = []
    total_shots = sum(counts.values())
    for element1 in povm_elements:
        for element2 in povm_elements:
            prob = counts.get('00', 0) * np.trace(np.kron(element1, element2) @ np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])) \
                 + counts.get('11', 0) * np.trace(np.kron(element1, element2) @ np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]))
            probabilities.append(prob / total_shots)
    
    return probabilities

# Define a noise model for simulation
noise_model = NoiseModel()
p_error_1q = 0.5  # Example error rate for 1-qubit gates
p_error_2q = 0.2  # Example error rate for 2-qubit gates
error_gate1 = depolarizing_error(p_error_1q, 1)
error_gate2 = depolarizing_error(p_error_2q, 2)
noise_model.add_all_qubit_quantum_error(error_gate1, ['x', 'h'])
noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])

# Step 2: Simulate POVM measurements
simulator = Aer.get_backend('aer_simulator')
shots = 1024
povm_elements = povm_tetrahedron()

# Simulate measurements for POVM elements
probabilities = simulate_povm(qc, povm_elements, simulator, shots)

# Step 3: Linear Inversion
start_time_li = time.perf_counter()

# Reconstruct the density matrix using POVM probabilities for two qubits
rho_li = sum(p * np.kron(E1, E2) for p, (E1, E2) in zip(probabilities, [(E1, E2) for E1 in povm_elements for E2 in povm_elements]))
rho_li = (rho_li + rho_li.T.conj()) / 2  # Make Hermitian
rho_li /= np.trace(rho_li)  # Normalize trace to 1

end_time_li = time.perf_counter()
runtime_li = end_time_li - start_time_li

# Define the ideal Bell state (|Φ+⟩ = (|00⟩ + |11⟩)/√2)
ideal_bell_state = np.array([[0.5, 0, 0, 0.5],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0.5, 0, 0, 0.5]])

# Convert matrices to Qiskit's DensityMatrix objects
rho_li_dm = DensityMatrix(rho_li)
ideal_bell_state_dm = DensityMatrix(ideal_bell_state)

# Calculate fidelity for Linear Inversion
fidelity_li = state_fidelity(rho_li_dm, ideal_bell_state_dm)

# Step 4: Maximum Likelihood Estimation (MLE)
def enforce_physical_density_matrix(rho):
    # Ensure the density matrix is positive semidefinite
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)  # Set negative eigenvalues to 0
    rho = (eigvecs @ np.diag(eigvals) @ eigvecs.T.conj())
    return rho / np.trace(rho)  # Normalize trace to 1

def likelihood(rho, povm_elements, probabilities):
    rho = rho.reshape((4, 4))  # Reshape the flat array back to matrix
    rho = (rho + rho.T.conj()) / 2  # Make Hermitian
    rho = enforce_physical_density_matrix(rho)  # Ensure physical state
    
    likelihood_value = 0
    for p, (E1, E2) in zip(probabilities, [(E1, E2) for E1 in povm_elements for E2 in povm_elements]):
        prob = np.real(np.trace(np.kron(E1, E2) @ rho))
        if prob > 0:
            likelihood_value += p * np.log(prob)
    return -likelihood_value  # Minimize the negative likelihood

# MLE Optimization
start_time_mle = time.perf_counter()

# Initial guess: start with the Linear Inversion density matrix
x0 = rho_li.flatten()

# Perform the optimization
result = minimize(likelihood, x0, args=(povm_elements, probabilities), method='Powell')
rho_mle = result.x.reshape((4, 4))

# Ensure the matrix is Hermitian and normalized
rho_mle = enforce_physical_density_matrix(rho_mle)

end_time_mle = time.perf_counter()
runtime_mle = end_time_mle - start_time_mle

# Convert to DensityMatrix object
rho_mle_dm = DensityMatrix(rho_mle)

# Calculate fidelity for MLE
fidelity_mle = state_fidelity(rho_mle_dm, ideal_bell_state_dm)

# Step 5: Print Results
print(f"Fidelity (Linear Inversion): {fidelity_li}, Runtime: {runtime_li:.4f} seconds")
print(f"Fidelity (MLE): {fidelity_mle}, Runtime: {runtime_mle:.4f} seconds")

# Step 6: Plot Fidelity and Runtime Comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Plot Fidelity
techniques = ['Linear Inversion', 'MLE']
fidelities = [fidelity_li, fidelity_mle]
ax1.plot(techniques, fidelities, label='Fidelity', marker='o', linestyle='--', color='blue')
ax1.set_ylabel('Fidelity')
ax1.set_title('Fidelity and Runtime Comparison: Linear Inversion vs MLE')
ax1.grid(True)

# Plot Runtime
runtimes = [runtime_li, runtime_mle]
ax2.plot(techniques, runtimes, label='Runtime (s)', marker='s', linestyle='--', color='orange')
ax2.set_ylabel('Runtime (s)')
ax2.grid(True)

# Show plots
plt.tight_layout()
plt.show()

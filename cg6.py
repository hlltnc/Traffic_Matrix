import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from scipy.optimize import minimize
from qiskit.visualization import plot_state_city
from qiskit.quantum_info import DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Create a Bell state circuit for two qubits
def create_bell_state_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Apply Hadamard gate to Alice's qubit
    qc.cx(0, 1)  # Apply CNOT gate to entangle Alice's and Bob's qubits
    return qc

# POVM elements (tetrahedral POVM)
def povm_tetrahedron():
    E0 = np.array([[1, 0], [0, 0]]) / 2  # POVM element 1
    E1 = np.array([[0, 0], [0, 1]]) / 2  # POVM element 2
    E2 = np.array([[0.5, 0.5], [0.5, 0.5]]) / 2  # POVM element 3
    E3 = np.array([[0.5, -0.5], [-0.5, 0.5]]) / 2  # POVM element 4
    return [E0, E1, E2, E3]

# Function to measure in a given basis (X, Y, Z, etc.)
def measure_in_basis(qc, basis):
    qc_copy = qc.copy()
    
    # Bob's measurement in the specified basis
    if basis == 'X':  # Measure in X basis
        qc_copy.h(1)  # Apply Hadamard gate to Bob's qubit
    elif basis == 'Y':  # Measure in Y basis
        qc_copy.sdg(1)  # Apply S-dagger gate to rotate to Y basis
        qc_copy.h(1)  # Apply Hadamard gate to Bob's qubit
    # If basis == 'Z', no need to add extra gates, it's the standard measurement basis
    
    qc_copy.measure([0, 1], [0, 1])  # Measure both Alice's and Bob's qubits
    return qc_copy

# Simulate POVM for a given circuit and N qubits
def simulate_povm(qc, povm_elements, simulator, shots):
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts()
    
    # Initialize probabilities for each POVM element
    probabilities = []
    total_shots = sum(counts.values())
    
    # Define the states that correspond to measurement outcomes for N qubits
    measurement_states = {
        '00': np.kron(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])),
        '11': np.kron(np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 1]]))
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
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.T.conj()
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

# Main Execution
povm_elements = povm_tetrahedron()  # Define the POVM elements (tetrahedral)

# Create the Bell state circuit
qc = create_bell_state_circuit()

# Simulate the POVM measurements in Z, X, and Y bases
simulator = Aer.get_backend('aer_simulator')
shots = 1024

# Measure in Z basis (standard computational basis)
qc_z = measure_in_basis(qc, 'Z')
probabilities_z = simulate_povm(qc_z, povm_elements, simulator, shots)

# Measure in X basis
qc_x = measure_in_basis(qc, 'X')
probabilities_x = simulate_povm(qc_x, povm_elements, simulator, shots)

# Measure in Y basis
qc_y = measure_in_basis(qc, 'Y')
probabilities_y = simulate_povm(qc_y, povm_elements, simulator, shots)

# Print the POVM probabilities for each basis
print("POVM Measurement Probabilities in Z basis:", probabilities_z)
print("POVM Measurement Probabilities in X basis:", probabilities_x)
print("POVM Measurement Probabilities in Y basis:", probabilities_y)

# MLE Optimization - Using the Z basis probabilities as an example for MLE reconstruction
initial_guess = np.eye(4) / 4  # Start with the maximally mixed state
rho_mle = mle_optimization(probabilities_z, povm_elements, initial_guess)

# Convert the density matrix to Qiskit DensityMatrix object for plotting
rho_mle_dm = DensityMatrix(rho_mle)

# Plot the MLE density matrix using plot_state_city
plot_state_city(rho_mle_dm, title="MLE Density Matrix")
plt.show()

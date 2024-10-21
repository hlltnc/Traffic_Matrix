import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute

# Create a Bell state circuit for four qubits (entangling two pairs)
def create_bell_state_circuit():
    qc = QuantumCircuit(4, 4)
    # First Bell pair
    qc.h(0)  # Apply Hadamard gate to the first qubit
    qc.cx(0, 1)  # Apply CNOT to entangle the first and second qubits
    # Second Bell pair
    qc.h(2)  # Apply Hadamard gate to the third qubit
    qc.cx(2, 3)  # Apply CNOT to entangle the third and fourth qubits
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
    
    # Apply measurements on qubits 1, 3 in specified basis (Bob and the third qubit)
    if basis == 'X':  # Measure in X basis
        qc_copy.h(1)  # Apply Hadamard gate to Bob's qubit
        qc_copy.h(3)  # Apply Hadamard gate to the fourth qubit
    elif basis == 'Y':  # Measure in Y basis
        qc_copy.sdg(1)  # Apply S-dagger gate to rotate to Y basis for Bob's qubit
        qc_copy.h(1)  # Apply Hadamard gate to Bob's qubit
        qc_copy.sdg(3)  # Apply S-dagger gate to rotate to Y basis for the fourth qubit
        qc_copy.h(3)  # Apply Hadamard gate to the fourth qubit
    # If basis == 'Z', no need to add extra gates, it's the standard measurement basis
    
    qc_copy.measure([0, 1, 2, 3], [0, 1, 2, 3])  # Measure all four qubits
    return qc_copy

# Simulate POVM for a given circuit and N qubits
def simulate_povm(qc, povm_elements, simulator, shots):
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts()
    
    # Initialize probabilities for each POVM element
    probabilities = []
    total_shots = sum(counts.values())
    
    # Define the states that correspond to measurement outcomes for 4 qubits
    measurement_states = {
        '0000': np.kron(np.kron(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])),
                        np.kron(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]]))),
        '1111': np.kron(np.kron(np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 1]])),
                        np.kron(np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 1]]))),
        '0101': np.kron(np.kron(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])),
                        np.kron(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))),
        '1010': np.kron(np.kron(np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
                        np.kron(np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])))
    }
    
    # Iterate over each POVM element combination and calculate probabilities
    for element1 in povm_elements:
        for element2 in povm_elements:
            prob = 0
            for key, state in measurement_states.items():
                prob += counts.get(key, 0) * np.real(np.trace(np.kron(np.kron(element1, element2), np.kron(element1, element2)) @ state))
            probabilities.append(prob / total_shots)
    
    return probabilities

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

# Plot the POVM probabilities for different bases
x_indices = np.arange(len(probabilities_z))  # Set up indices for x-axis
bar_width = 0.2  # Adjust for spacing

plt.bar(x_indices, probabilities_z, width=bar_width, color='blue', label='Z basis', alpha=0.7)
plt.bar(x_indices + bar_width, probabilities_x, width=bar_width, color='orange', label='X basis', alpha=0.7)
plt.bar(x_indices + 2 * bar_width, probabilities_y, width=bar_width, color='green', label='Y basis', alpha=0.7)

plt.xlabel('POVM Outcome Index')
plt.ylabel('Probability')
plt.title('POVM Measurement Probabilities in Various Bases')
plt.legend()
plt.grid(True)
plt.show()

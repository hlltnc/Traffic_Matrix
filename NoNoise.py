import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit,transpile
from qiskit_aer import Aer
# Create a circuit that generates 2 Bell states (4 qubits total)
def create_bell_state_circuit():
    qc = QuantumCircuit(4, 4)  # 4 qubits, 4 classical bits
    # First Bell pair (qubits 0 and 1)
    qc.h(0)  # Apply Hadamard gate to the first qubit
    qc.cx(0, 1)  # Apply CNOT to entangle the first and second qubits
    # Second Bell pair (qubits 2 and 3)
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
    # Apply measurements on the second qubit of each Bell pair
    for i in [1, 3]:  # These are Bob's qubits
        if basis == 'X':  # Measure in X basis
            qc_copy.h(i)  # Apply Hadamard gate
        elif basis == 'Y':  # Measure in Y basis
            qc_copy.sdg(i)  # Apply S-dagger gate
            qc_copy.h(i)  # Apply Hadamard gate
        # If basis == 'Z', no need to add extra gates, it's the standard measurement basis
    # Measure all 4 qubits
    qc_copy.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return qc_copy
# Memory-efficient Kronecker product computation
def progressive_kron(elements):
    kron_prod = elements[0]
    for el in elements[1:]:
        kron_prod = np.kron(kron_prod, el)
    return kron_prod
# Simulate POVM for a given circuit and N qubits
def simulate_povm(qc, povm_elements, simulator, shots):

    esult=transpile(qc, simulator)
    
    job = simulator.run(qc, shots=1024)  # Fazladan simulator argümanı kaldırıldı

    result = job.result()
    counts = result.get_counts()


    # Initialize probabilities for each POVM element
    probabilities = []
    total_shots = sum(counts.values())
    # Generate all possible 4^4 = 256 POVM outcomes for 4 qubits
    povm_indices = np.arange(256)  # Range for 4^4 possibilities
    # Iterate over each POVM outcome combination and calculate probabilities
    for idx in povm_indices:  # Loop through all 256 possible outcome indices
        povm_key = np.base_repr(idx, base=4).zfill(4)  # Base 4 representation
        # Extract the corresponding POVM elements for each qubit
        povm_element_combination = [povm_elements[int(povm_key[i])] for i in range(4)]
        # Calculate the Kronecker product for this combination (progressively)
        povm_combined = progressive_kron(povm_element_combination)
        # Calculate the probability based on measurement outcomes
        prob = 0
        for outcome, count in counts.items():
            outcome_state = np.array([[1, 0], [0, 0]]) if outcome[0] == '0' else np.array([[0, 0], [0, 1]])
            for q in range(1, 4):
                outcome_state = np.kron(outcome_state, np.array([[1, 0], [0, 0]]) if outcome[q] == '0' else np.array([[0, 0], [0, 1]]))
            prob += count * np.real(np.trace(povm_combined @ outcome_state))
        probabilities.append(prob / total_shots)
    return probabilities
# Main Execution
povm_elements = povm_tetrahedron()  # Define the POVM elements (tetrahedral)
# Create the Bell state circuit with 2 Bell states (4 qubits total)
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
# print("POVM Measurement Probabilities in Z basis:", probabilities_z)
# print("POVM Measurement Probabilities in X basis:", probabilities_x)
# print("POVM Measurement Probabilities in Y basis:", probabilities_y)
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
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import partial_trace
from scipy.linalg import logm


import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_state_city



# Function to calculate von Neumann entropy
def von_neumann_entropy(rho):
    # Calculate log2 of rho
    rho_log = logm(rho) / np.log(2)
    # Calculate -Tr(ρ log2 ρ)
    entropy = -np.trace(np.dot(rho, rho_log)).real
    return entropy

# Function to calculate Ig(Φy) = S(ρo) - S(ρio)
def quantum_channel_capacity(rho_o, rho_io, numb_of_uses):
    S_rho_o = von_neumann_entropy(rho_o)
    S_rho_io = von_neumann_entropy(rho_io)
    # Calculate the channel capacity
    Channel_Capacity = S_rho_o - S_rho_io / numb_of_uses
    return Channel_Capacity

# Create a Bell pair (entangled qubits)
def create_bell_pair():
    qc = QuantumCircuit(4)  # Create 4 qubits to work with Bell pairs
    qc.h(0)  # Apply Hadamard gate to the first qubit (Alice's qubit)
    qc.cx(0, 1)  # Apply CNOT to entangle qubits (creates a Bell state)
    # Second Bell pair (qubits 2 and 3)
    qc.h(2)
    qc.cx(2, 3)
    
    return qc

# Function to create the full density matrix for both qubits (ρo)
def get_full_density_matrix():
    # Here, replace with your actual tomography code
    import State_Tomog_4qubit
    import Tomog_NoNoise

    circuits = State_Tomog_4qubit.generate_tomography_circuits()
    counts = State_Tomog_4qubit.perform_tomography(circuits, shots=1024)
    rho = State_Tomog_4qubit.reconstruct_density_matrix(counts, shots=1024)
    
    rho_reconstructed_dm = DensityMatrix(rho)
    rho_full = rho_reconstructed_dm

    circuits_NoNoise = Tomog_NoNoise.generate_tomography_circuits()
    counts_NoNoise = Tomog_NoNoise.perform_tomography(circuits_NoNoise, shots=1024)
    rho_NoNoise = Tomog_NoNoise.reconstruct_density_matrix(counts_NoNoise, shots=1024)
    
    rho_reconstructed_dm_NoNoise = DensityMatrix(rho_NoNoise)

    rho_o = rho_reconstructed_dm
    rho_io = rho_reconstructed_dm_NoNoise

    return rho_io, rho_o

# Main code
rho_io, rho_o = get_full_density_matrix()

# Create an empty list to store channel capacities for different uses
Channel_Capacity_list = []

numb_of_uses_list = list(range(1, 50))

# Loop over different numbers of uses



# Loop over different numbers of uses
for numb_of_uses in numb_of_uses_list:
    # Calculate the quantum channel capacity C(Φy)
    capacity = quantum_channel_capacity(rho_o, rho_io, numb_of_uses)
    Channel_Capacity_list.append(capacity)

# Output the results
#print(f"Bob's reduced density matrix (ρio):\n{rho_io}")
#print(f"Full system density matrix (ρo):\n{rho_o}")
print(f"Quantum channel capacity list: C(Φy) = {Channel_Capacity_list}")


import cap_ent_4qubit2BitFlip

rho_io, rho_o = cap_ent_4qubit2BitFlip.get_full_density_matrix()

Channel_Capacity_list_bitFlip = []

numb_of_uses_list = list(range(1, 50))

# Loop over different numbers of uses
for numb_of_uses in numb_of_uses_list:
    # Calculate the quantum channel capacity C(Φy)
    capacity = cap_ent_4qubit2BitFlip.quantum_channel_capacity(rho_o, rho_io, numb_of_uses)
    Channel_Capacity_list_bitFlip.append(capacity)

# Output the results
#print(f"Bob's reduced density matrix (ρio):\n{rho_io}")
#print(f"Full system density matrix (ρo):\n{rho_o}")
print(f"Quantum channel capacity list Bit Flip: C(Φy) = {Channel_Capacity_list_bitFlip}")


plt.plot(numb_of_uses_list, Channel_Capacity_list, marker='o', linestyle='-', color='b', label='depolarizing channel')
plt.plot(numb_of_uses_list, Channel_Capacity_list_bitFlip, marker='s', linestyle='--', color='r', label='BitFlip Channel')

# Grafik başlıkları ve eksen adları
#plt.title('Two Plots in the Same Chart')
plt.xlabel('number of channel uses')
plt.ylabel('channel capacity')

# Grafikteki çizgilere ait etiketleri göstermek için
plt.legend()

# Grid eklemek için
plt.grid(True)

# Grafiği göster
plt.show()









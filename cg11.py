import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Choi, Operator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.stats import unitary_group

from qiskit import __version__
print(__version__)
# Function to generate a random quantum channel (random unitary + noise)
def generate_random_channel(N=1):
    dim = 2 ** N
    # Generate a random unitary operator using scipy
    U = unitary_group.rvs(dim)
    
    # Create a random depolarizing noise channel
    p = np.random.uniform(0, 0.5)  # Random depolarizing probability
    noise = (1 - p) * np.eye(dim) + p * U

    return Operator(noise)
# Calculate channel capacity (upper bound on capacity) for random channels with refined eigenvalue filtering
# Calculate channel capacity (upper bound on capacity) for random channels with stricter handling
def calculate_channel_capacity(channel):
    choi = Choi(channel)
    eigenvalues = np.linalg.eigvalsh(choi.data)  # Hermitian eigenvalue calculation (eigenvalsh)
    
    # Debug: Print eigenvalues for inspection
    print("Eigenvalues of the Choi matrix:", eigenvalues)

    # Filter out small or negative eigenvalues (set negative values to zero)
    valid_eigenvalues = [ev for ev in eigenvalues if ev > 1e-5]  # Filter out small and negative eigenvalues

    if not valid_eigenvalues:  # If no valid eigenvalues, capacity is 0
        return 0

    # Compute channel capacity using the valid eigenvalues
    channel_capacity = np.real(np.sum([-ev * np.log2(ev) for ev in valid_eigenvalues]))
    
    # If the channel capacity is negative, set it to 0 (as capacity should not be negative)
    if channel_capacity < 0:
        channel_capacity = 0

    # Debug: Print channel capacity after calculation
    print("Calculated channel capacity:", channel_capacity)
    
    return channel_capacity




# Number of uses of the channel
channel_uses = range(1, 11)  # Simulate from 1 to 10 uses of the channel
capacities = []

# Simulate the capacity for different uses of the channel
for num_uses in channel_uses:
    # Generate a random channel
    channel = generate_random_channel()
    
    # Calculate the channel capacity for a single use
    capacity_per_use = calculate_channel_capacity(channel)
    
    # Assume the capacity scales linearly with the number of uses
    total_capacity = capacity_per_use * num_uses
    capacities.append(total_capacity)

# Plotting Channel Capacity vs Number of Uses of the Channel
plt.plot(channel_uses, capacities, marker='o', linestyle='--', color='blue', label='Channel Capacity')
plt.title('Channel Capacity vs Number of Uses of the Channel')
plt.xlabel('Number of Uses of the Channel')
plt.ylabel('Channel Capacity (bits)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

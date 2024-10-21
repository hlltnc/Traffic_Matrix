import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Choi, Operator

# Function to generate a fixed unitary depolarizing quantum channel with noise
def generate_fixed_unitary_channel(N=1, p=0.1):
    dim = 2 ** N
    # Use identity as the fixed unitary (no randomness)
    U = np.eye(dim)
    
    # Apply depolarizing noise
    noise = (1 - p) * np.eye(dim) + p * U
    return Operator(noise)

# Calculate channel capacity (upper bound on capacity) for noisy channels
def calculate_channel_capacity(channel):
    choi = Choi(channel)
    eigenvalues = np.linalg.eigvalsh(choi.data)  # Hermitian eigenvalue calculation (eigenvalsh)
    
    # Filter out small or negative eigenvalues (set negative values to zero)
    valid_eigenvalues = [ev for ev in eigenvalues if ev > 1e-5]  # Filter out small and negative eigenvalues

    if not valid_eigenvalues:  # If no valid eigenvalues, capacity is 0
        return 0

    # Compute channel capacity using the valid eigenvalues
    channel_capacity = np.real(np.sum([-ev * np.log2(ev) for ev in valid_eigenvalues]))
    
    # If the channel capacity is negative, set it to 0 (as capacity should not be negative)
    if channel_capacity < 0:
        channel_capacity = 0

    return channel_capacity

# Range of noise strengths (from 0 to 1)
noise_strengths = np.linspace(0, 1, 20)  # Varying the noise strength from 0 to 1
capacities = []

# Simulate the channel capacity for different noise strengths with a fixed unitary
for p in noise_strengths:
    # Generate a noisy channel with the given noise strength p and fixed unitary
    channel = generate_fixed_unitary_channel(N=1, p=p)
    
    # Calculate the channel capacity
    capacity = calculate_channel_capacity(channel)
    capacities.append(capacity)

# Plotting Channel Capacity vs Noise Strength
plt.plot(noise_strengths, capacities, marker='o', linestyle='--', color='blue', label='Channel Capacity')
plt.title('Channel Capacity vs Noise Strength (Fixed Unitary)')
plt.xlabel('Noise Strength (p)')
plt.ylabel('Channel Capacity (bits)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

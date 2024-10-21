import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Choi, partial_trace, Operator, random_unitary
from scipy.optimize import minimize
import numpy as np
from scipy.stats import unitary_group

# Function to calculate success probability of distinguishing two channels
def success_probability(channel1, channel2, rho):
    # Create Choi states for each channel
    choi1 = Choi(channel1)
    choi2 = Choi(channel2)

    # Create the difference between the two channels (trace distance)
    diff = choi1.data - choi2.data

    # Trace distance is proportional to the distinguishability
    trace_dist = np.trace(np.abs(diff))

    # Success probability is related to trace distance by this formula
    success_prob = 0.5 * (1 + trace_dist / 2)
    return success_prob

# Function to generate a random quantum channel (random unitary + noise)



# Function to generate a random unitary matrix using scipy
def random_unitary(dim):
    return unitary_group.rvs(dim)

# Modify the original channel generator
def generate_random_channel(N=1):
    dim = 2 ** N
    # Generate a random unitary operator using scipy
    U = random_unitary(dim)
    
    # Create a random depolarizing noise channel
    p = np.random.uniform(0, 0.5)  # Random depolarizing probability
    noise = (1 - p) * np.eye(dim) + p * U

    return noise
# Calculate channel capacity (upper bound on capacity) for random channels
def calculate_channel_capacity(channel):
    choi = Choi(channel)
    eigenvalues = np.linalg.eigvalsh(choi.data)  # Hermitian eigenvalue calculation (eigenvalsh)
    
    # Debug: Print eigenvalues to check them
    print("Eigenvalues of the Choi matrix:", eigenvalues)

    # Ensure only positive eigenvalues are included in the capacity calculation
    valid_eigenvalues = [ev for ev in eigenvalues if ev > 0]

    if not valid_eigenvalues:  # If no valid eigenvalues, capacity is 0
        return 0

    # Compute channel capacity using the valid eigenvalues
    # Capacity formula is -sum(ev * log2(ev)) for non-zero ev
    channel_capacity = np.real(np.sum([-ev * np.log2(ev) for ev in valid_eigenvalues]))
    return channel_capacity


    #print("Eigenvalues of the Choi matrix:", eigenvalues)

# Generate data for plotting
success_probabilities = []
channel_capacities = []

N_samples = 50  # Number of random channel samples
for _ in range(N_samples):
    # Generate two random channels
    channel1 = generate_random_channel()
    channel2 = generate_random_channel()
    
    # Use an input state (can be a maximally mixed state or other)
    input_state = np.array([[0.5, 0], [0, 0.5]])  # Example mixed state

    # Calculate success probability
    success_prob = success_probability(channel1, channel2, input_state)
    success_probabilities.append(success_prob)
    
    # Calculate channel capacity (for one of the channels)
    capacity = calculate_channel_capacity(channel1)
    channel_capacities.append(capacity)

#print("Eigenvalues of the Choi matrix:", eigenvalues)

# Plot Channel Capacity vs Success Probability
plt.scatter(channel_capacities, success_probabilities, color='blue', label="Channel Capacity vs Success Prob")
plt.title('Channel Capacity vs Success Probability of Distinguishing Channels')
plt.xlabel('Channel Capacity (bits)')
plt.ylabel('Success Probability')
plt.grid(True)
plt.legend()
plt.show()

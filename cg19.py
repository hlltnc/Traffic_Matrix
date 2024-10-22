import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Define the tomography techniques from previous code (MLE, LRE, BME)
# Improved Maximum Likelihood Estimation (MLE) method with validation checks
def reconstruct_mle(counts, num_qubits):
    dim = 2 ** num_qubits

    # Define a likelihood function for optimization
    def likelihood_function(rho_flat, counts):
        rho = rho_flat.reshape((dim, dim))
        rho = (rho + rho.T.conj()) / 2  # Ensure Hermiticity
        rho /= np.trace(rho)  # Normalize trace to 1

        log_likelihood = 0
        total_counts = sum(counts.values())
        for outcome, count in counts.items():
            # Convert binary outcome string to integer index
            idx = int(outcome.replace(' ', ''), 2)
            prob = np.real(np.trace(rho @ np.outer(np.eye(dim)[idx], np.eye(dim)[idx])))  # Probability of outcome
            if prob > 0:
                log_likelihood += count * np.log(prob)
            else:
                log_likelihood += count * np.log(1e-10)  # To avoid log(0)
        return -log_likelihood  # Maximize log-likelihood

    # Define a function to enforce positive semi-definiteness and trace=1
    def enforce_density_matrix_constraints(rho_flat):
        rho = rho_flat.reshape((dim, dim))
        rho = (rho + rho.T.conj()) / 2  # Ensure Hermiticity
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)  # Force non-negative eigenvalues
        rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T.conj()
        rho /= np.trace(rho)  # Normalize trace to 1
        return rho.flatten()

    # Use LRE as the starting point for MLE
    rho_lre = reconstruct_lre(counts, num_qubits)
    rho_init_flat = rho_lre.flatten()

    # Optimize the likelihood function
    result = minimize(likelihood_function, rho_init_flat, args=(counts,), method='L-BFGS-B', 
                      options={'maxiter': 1000})

    # Reshape the result back into a matrix and enforce constraints
    rho_mle_flat = result.x
    rho_mle_flat = enforce_density_matrix_constraints(rho_mle_flat)
    rho_mle = rho_mle_flat.reshape((dim, dim))

    # Validation checks
    validate_density_matrix(rho_mle)

    return rho_mle

# Function to validate the density matrix
def validate_density_matrix(rho):
    """
    Validates if the given density matrix is:
    1. Hermitian
    2. Positive semi-definite
    3. Trace equals to 1
    """
    # Check if the matrix is Hermitian
    if not np.allclose(rho, rho.T.conj()):
        raise ValueError("Density matrix is not Hermitian")

    # Check if the matrix is positive semi-definite (all eigenvalues should be >= 0)
    eigenvalues = np.linalg.eigvals(rho)
    if np.any(eigenvalues < 0):
        raise ValueError("Density matrix has negative eigenvalues")

    # Check if the trace of the matrix is 1
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Density matrix trace is not equal to 1")

    print("Density matrix is valid.")

# Linear Regression Estimation (LRE)
def reconstruct_lre(counts, num_qubits):
    dim = 2 ** num_qubits
    rho = np.zeros((dim, dim), dtype=complex)  # Initialize density matrix

    # Example: Use counts to estimate the diagonal elements of the density matrix
    total_counts = sum(counts.values())  # Total number of measurement shots

    # Fill in the diagonal elements based on measurement outcomes
    for outcome, count in counts.items():
        prob = count / total_counts  # Probability of the outcome
        idx = int(outcome.replace(' ', ''), 2)  # Convert bitstring outcome to an integer index
        rho[idx, idx] = prob  # Set diagonal elements based on probabilities

    # Ensure trace is 1 (it should already be 1 if correctly filled, but normalize to be sure)
    rho /= np.trace(rho)

    # Ensure Hermiticity (LRE may not produce a perfectly Hermitian matrix, so enforce it)
    rho = (rho + rho.T.conj()) / 2

    # Ensure positive semi-definiteness (all eigenvalues should be non-negative)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)  # Force non-negative eigenvalues
    rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T.conj()

    return rho

# Bayesian Mean Estimation (BME)
def reconstruct_bme(counts, num_qubits):
    dim = 2 ** num_qubits
    rho_bme = np.eye(dim) / dim  # Start with the maximally mixed state as the prior
    
    # Total number of shots
    total_counts = sum(counts.values())
    
    # Update the density matrix based on measurement counts
    for outcome, count in counts.items():
        # Convert binary outcome string to an integer index
        idx = int(outcome.replace(' ', ''), 2)
        
        # Calculate the weight of this outcome based on the number of occurrences
        weight = count / total_counts
        
        # Update the BME density matrix by adding the contribution of this outcome
        # Construct the pure state corresponding to this outcome and update the density matrix
        projector = np.outer(np.eye(dim)[idx], np.eye(dim)[idx])
        rho_bme = (1 - weight) * rho_bme + weight * projector
    
    # Normalize trace to ensure it's a valid density matrix
    rho_bme /= np.trace(rho_bme)
    
    return rho_bme


# Function to time a tomography method
def time_tomography(tomography_func, counts, num_qubits):
    start_time = time.time()
    tomography_func(counts, num_qubits)
    end_time = time.time()
    return end_time - start_time

# Function to simulate counts for a given number of qubits
def simulate_counts(num_qubits, shots=1024):
    dim = 2 ** num_qubits
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Create Bell states for all pairs (or simple entangled states)
    for i in range(0, num_qubits, 2):
        circuit.h(i)
        if i+1 < num_qubits:
            circuit.cx(i, i+1)
    
    # Add measurements to all qubits
    circuit.measure(range(num_qubits), range(num_qubits))
    
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, simulator, shots=shots).result()
    counts = result.get_counts()
    return counts

# Plotting the complexity in terms of number of qubits
def plot_complexity(max_qubits=6, shots=1024):
    num_qubits_list = list(range(2, max_qubits + 1, 2))  # Evaluate complexity from 2 to max_qubits
    times_mle = []
    times_lre = []
    times_bme = []

    for num_qubits in num_qubits_list:
        print(f"Running tomography for {num_qubits} qubits...")

        # Simulate measurement counts for the given number of qubits
        counts = simulate_counts(num_qubits, shots=shots)

        # Measure time for MLE, LRE, BME
        time_mle = time_tomography(reconstruct_mle, counts, num_qubits)
        time_lre = time_tomography(reconstruct_lre, counts, num_qubits)
        time_bme = time_tomography(reconstruct_bme, counts, num_qubits)

        # Store the times
        times_mle.append(time_mle)
        times_lre.append(time_lre)
        times_bme.append(time_bme)

    # Plot the results
    plt.plot(num_qubits_list, times_mle, label="MLE")
    plt.plot(num_qubits_list, times_lre, label="LRE")
    plt.plot(num_qubits_list, times_bme, label="BME")

    plt.xlabel("Number of Qubits")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Complexity of MLE, LRE, and BME in Terms of Number of Qubits")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run and plot complexity for 2 to 6 qubits (adjust max_qubits for higher qubits)
plot_complexity(max_qubits=6)

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize

# Define constants
num_qubits = 4
dim = 2 ** num_qubits

# Create a circuit that generates 2 Bell states (4 qubits total)
def create_bell_state_circuit(with_measurements=True):
    qc = QuantumCircuit(4, 4)  # 4 qubits, 4 classical bits for measurement
    # First Bell pair (qubits 0 and 1)
    qc.h(0)
    qc.cx(0, 1)
    # Second Bell pair (qubits 2 and 3)
    qc.h(2)
    qc.cx(2, 3)
    
    if with_measurements:
        # Add measurements to all qubits
        qc.measure([0, 1, 2, 3], [0, 1, 2, 3])  # Measure qubits 0-3 into classical bits 0-3
    
    return qc

# Add depolarizing noise to the circuit
def add_depolarizing_noise():
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.01, 1)  # 1% error rate for single-qubit gates
    error_2q = depolarizing_error(0.01, 2)  # 2% error rate for two-qubit gates
    noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model

# Perform POVM measurements
def perform_povm_measurements(circuit, shots=1024):
    simulator = Aer.get_backend('qasm_simulator')
    noise_model = add_depolarizing_noise()
    # Execute the circuit on the simulator
    result = execute(circuit, simulator, noise_model=noise_model, shots=shots).result()
    
    # Retrieve the counts for the first (and only) circuit in the result
    counts = result.get_counts(0)
    print("Measurement counts:", counts)
    return counts

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


# Main experiment loop to compute fidelity over time (steps)
def run_experiment(steps=50, max_shots=1024):
    # Create Bell state circuit WITHOUT measurements for the ideal density matrix
    circuit_no_measurements = create_bell_state_circuit(with_measurements=False)
    ideal_density_matrix = DensityMatrix.from_instruction(circuit_no_measurements)

    # Create Bell state circuit WITH measurements for POVM simulation
    circuit_with_measurements = create_bell_state_circuit(with_measurements=True)

    # Arrays to store fidelities over time
    fidelity_mle = []
    fidelity_lre = []
    fidelity_bme = []
    
    for step in range(1, steps + 1):
        shots = max_shots * step // steps  # Increase shots over time
        counts = perform_povm_measurements(circuit_with_measurements, shots=shots)

        # Reconstruct states using MLE, LRE, BME
        rho_mle = reconstruct_mle(counts, num_qubits)
        rho_lre = reconstruct_lre(counts, num_qubits)
        rho_bme = reconstruct_bme(counts, num_qubits)

        # Compute fidelities for each method
        fidelity_mle_value = state_fidelity(ideal_density_matrix, rho_mle)
        fidelity_lre_value = state_fidelity(ideal_density_matrix, rho_lre)
        fidelity_bme_value = state_fidelity(ideal_density_matrix, rho_bme)

        # Print the fidelities for debugging
        print(f"Step {step}:")
        print(f"  MLE Fidelity: {fidelity_mle_value}")
        print(f"  LRE Fidelity: {fidelity_lre_value}")
        print(f"  BME Fidelity: {fidelity_bme_value}")

        # Append fidelities to arrays
        fidelity_mle.append(fidelity_mle_value)
        fidelity_lre.append(fidelity_lre_value)
        fidelity_bme.append(fidelity_bme_value)

    return fidelity_mle, fidelity_lre, fidelity_bme

# Plot fidelities over time
def plot_fidelity(fidelity_mle, fidelity_lre, fidelity_bme):
    steps = len(fidelity_mle)
    x = np.linspace(1, steps, steps)

    plt.plot(x, fidelity_mle, label="MLE")
    plt.plot(x, fidelity_lre, label="LRE")
    plt.plot(x, fidelity_bme, label="BME")
    #plt.yscale("log")   
    plt.xlabel("Time (second)")
    plt.ylabel("Fidelity")
    #plt.title("Fidelity of MLE, LRE, BME over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig('plotmlb.pdf')
    plt.show()

# Run the experiment and plot results
fidelity_mle, fidelity_lre, fidelity_bme = run_experiment(steps=50)
plot_fidelity(fidelity_mle, fidelity_lre, fidelity_bme)

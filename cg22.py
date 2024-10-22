import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import DensityMatrix
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize
import warnings
from scipy._lib._testutils import PytestTester  # This prevents unnecessary complex warnings in SciPy

# Suppress complex number warnings from SciPy
warnings.filterwarnings("ignore", category=RuntimeWarning)
def add_depolarizing_noise():
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.6, 1)  # 1% error rate for single-qubit gates
    error_2q = depolarizing_error(0.7, 2)  # 2% error rate for two-qubit gates
    noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model
# Function to simulate bit-flip noise
# def add_bit_flip_noise():
#     noise_model = NoiseModel()
#     error_1q = pauli_error([('X', 0.05), ('I', 0.95)])  # 5% chance of bit-flip, 95% no error
#     noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x'])
#     error_2q = pauli_error([('XX', 0.05), ('II', 0.95)])  # 5% chance of bit-flip on both qubits
#     noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
#     return noise_model

# Function to create Bell state circuits for N qubits
def create_bell_state_circuit(num_qubits, with_measurements=True):
    qc = QuantumCircuit(num_qubits, num_qubits)
    for i in range(0, num_qubits, 2):
        qc.h(i)
        if i + 1 < num_qubits:
            qc.cx(i, i + 1)
    if with_measurements:
        qc.measure(list(range(num_qubits)), list(range(num_qubits)))
    return qc

# Maximum Likelihood Estimation (MLE)
def reconstruct_mle(counts, num_qubits):
    dim = 2 ** num_qubits

    def likelihood_function(rho_flat, counts):
        rho = rho_flat.reshape((dim, dim))
        rho = (rho + rho.T.conj()) / 2  # Ensure Hermiticity
        rho /= np.real(np.trace(rho))  # Normalize trace to 1

        log_likelihood = 0
        total_counts = sum(counts.values())
        for outcome, count in counts.items():
            idx = int(outcome.replace(' ', ''), 2)
            prob = np.real(np.trace(rho @ np.outer(np.eye(dim)[idx], np.eye(dim)[idx])))
            if prob > 0:
                log_likelihood += count * np.log(prob)
            else:
                log_likelihood += count * np.log(1e-10)  # To avoid log(0)
        return -log_likelihood  # Maximize log-likelihood

    def enforce_density_matrix_constraints(rho_flat):
        rho = rho_flat.reshape((dim, dim))
        rho = (rho + rho.T.conj()) / 2  # Ensure Hermiticity
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)  # Force non-negative eigenvalues
        rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T.conj()
        rho /= np.real(np.trace(rho))  # Normalize trace to 1
        return rho.flatten()

    rho_init_flat = np.eye(dim).flatten()  # Start with maximally mixed state as the initial guess

    result = minimize(likelihood_function, rho_init_flat, args=(counts,), method='L-BFGS-B', 
                      options={'maxiter': 1000})

    rho_mle_flat = result.x
    rho_mle_flat = enforce_density_matrix_constraints(rho_mle_flat)
    rho_mle = rho_mle_flat.reshape((dim, dim))
    
    validate_density_matrix(rho_mle)
    return rho_mle

# Linear Regression Estimation (LRE)
def reconstruct_lre(counts, num_qubits):
    dim = 2 ** num_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    total_counts = sum(counts.values())

    for outcome, count in counts.items():
        prob = count / total_counts
        idx = int(outcome.replace(' ', ''), 2)
        rho[idx, idx] = prob

    rho /= np.real(np.trace(rho))
    rho = (rho + rho.T.conj()) / 2  # Ensure Hermiticity
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)  # Force non-negative eigenvalues
    rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T.conj()

    return rho

# Bayesian Mean Estimation (BME)
def reconstruct_bme(counts, num_qubits):
    dim = 2 ** num_qubits
    rho_bme = np.eye(dim) / dim
    total_counts = sum(counts.values())

    for outcome, count in counts.items():
        idx = int(outcome.replace(' ', ''), 2)
        weight = count / total_counts
        projector = np.outer(np.eye(dim)[idx], np.eye(dim)[idx])
        rho_bme = (1 - weight) * rho_bme + weight * projector

    rho_bme /= np.real(np.trace(rho_bme))
    return rho_bme

# Function to validate the density matrix
def validate_density_matrix(rho):
    if not np.allclose(rho, rho.T.conj()):
        raise ValueError("Density matrix is not Hermitian")
    eigenvalues = np.linalg.eigvals(rho)
    if np.any(eigenvalues < 0):
        raise ValueError("Density matrix has negative eigenvalues")
    if not np.isclose(np.real(np.trace(rho)), 1):
        raise ValueError("Density matrix trace is not equal to 1")
    print("Density matrix is valid.")

# Function to measure time of tomography method
def time_tomography_method(method_func, counts, num_qubits):
    start_time = time.perf_counter()  # More precise timing for fast functions
    method_func(counts, num_qubits)
    end_time = time.perf_counter()
    return end_time - start_time

# Function to simulate counts for a given number of qubits
def simulate_counts(num_qubits, shots=8192):
    circuit = create_bell_state_circuit(num_qubits, with_measurements=True)
    simulator = Aer.get_backend('qasm_simulator')
    noise_model = add_depolarizing_noise()
    result = execute(circuit, simulator, noise_model=noise_model, shots=shots).result()
    return result.get_counts()

# Function to plot running time as a function of the number of qubits
def plot_running_time(max_qubits=6, shots=8192):
    num_qubits_list = list(range(2, max_qubits + 1, 2))  # Test for 2 to max_qubits qubits
    times_mle = []
    times_lre = []
    times_bme = []

    for num_qubits in num_qubits_list:
        print(f"Running tomography for {num_qubits} qubits...")

        # Simulate measurement counts
        counts = simulate_counts(num_qubits, shots=shots)

        # Measure execution time for MLE, LRE, and BME
        time_mle = time_tomography_method(reconstruct_mle, counts, num_qubits)
        time_lre = time_tomography_method(reconstruct_lre, counts, num_qubits)
        time_bme = time_tomography_method(reconstruct_bme, counts, num_qubits)

        # Append times to lists
        times_mle.append(time_mle)
        times_lre.append(time_lre)
        times_bme.append(time_bme)

    # Plot the running


    # Plot the running time
    plt.plot(num_qubits_list, times_mle, label="MLE", marker='o')
    plt.plot(num_qubits_list, times_lre, label="LRE", marker='x')
    plt.plot(num_qubits_list, times_bme, label="BME", marker='s')
    plt.yscale('log')
    plt.xlabel("Number of Qubits")
    plt.ylabel("Time Complexity (second)")
   # plt.title("Running Time vs. Number of Qubits")
    plt.legend()
    plt.grid(True)
    plt.savefig('running_time_plot2.pdf')
    plt.show()

# Run and plot the running time
plot_running_time(max_qubits=6)

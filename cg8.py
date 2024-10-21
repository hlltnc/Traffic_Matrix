import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter
from scipy.optimize import minimize

# Function to generate Bell state circuits with variable N (number of qubits)
def create_bell_circuit(N):
    qc = QuantumCircuit(N, N)  # N qubits and N classical bits
    qc.h(0)  # Apply Hadamard gate to the first qubit
    for i in range(N-1):
        qc.cx(i, i+1)  # Apply CNOT gates to entangle
    qc.measure(range(N), range(N))  # Add measurement gates
    return qc

# Simulate POVM for a given circuit and N qubits
def simulate_povm(qc, simulator, shots):
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts()
    return counts

# Apply measurement error mitigation to counts
def apply_meas_mitigation(counts, meas_fitter):
    mitigated_counts = meas_fitter.filter.apply(counts)
    return mitigated_counts

# Noise model
noise_model = NoiseModel()
p_error_1q = 0.05
p_error_2q = 0.02
error_gate1 = depolarizing_error(p_error_1q, 1)
error_gate2 = depolarizing_error(p_error_2q, 2)
noise_model.add_all_qubit_quantum_error(error_gate1, ['x', 'h'])
noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])

# Simulator
simulator = Aer.get_backend('aer_simulator')
shots = 1024

# Run the comparison between Linear Inversion and MLE for different N
N_values = list(range(2, 6))  # Example N values (number of qubits)
fidelity_li_list = []
fidelity_mle_list = []
runtime_li_list = []
runtime_mle_list = []

# Ideal Bell state for comparison (for two qubits)
ideal_bell_state = np.array([[0.5, 0, 0, 0.5],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0.5, 0, 0, 0.5]])

# Loop over different values of N
for N in N_values:
    # Create the Bell circuit
    qc = create_bell_circuit(N)
    
    # Generate the measurement calibration circuits for the correct number of qubits
    meas_cal_circuits, state_labels = complete_meas_cal(qr=qc.qregs[0])
    meas_cal_results = execute(meas_cal_circuits, backend=simulator, shots=shots).result()
    meas_fitter = CompleteMeasFitter(meas_cal_results, state_labels)

    # Simulate POVM measurements
    counts = simulate_povm(qc, simulator, shots)
    
    # Apply measurement error mitigation
    mitigated_counts = apply_meas_mitigation(counts, meas_fitter)
    
    # Debug: print raw and mitigated counts
    print(f"N = {N}")
    print(f"Raw counts: {counts}")
    print(f"Mitigated counts: {mitigated_counts}")

    # (Placeholders for fidelity calculations after processing the mitigated counts)
    # You would follow the rest of the logic to reconstruct the density matrix and calculate the fidelities as per your existing code.
    
    # Store results (for illustrative purposes)
    fidelity_li_list.append(np.random.rand())  # Replace with real fidelity from Linear Inversion
    fidelity_mle_list.append(np.random.rand())  # Replace with real fidelity from MLE

# Plotting fidelity for different N
fig, ax1 = plt.subplots()

# Plot fidelity
ax1.plot(N_values, fidelity_li_list, label='Linear Regression', marker='o', linestyle='--', color='blue')
ax1.plot(N_values, fidelity_mle_list, label='MLE', marker='s', linestyle='--', color='orange')
ax1.set_ylabel('Fidelity')
ax1.set_xlabel('N (number of qubits)')
ax1.set_title('Fidelity Comparison for Linear Regression and MLE (Error Mitigation)')
ax1.legend()
ax1.grid(True)

# Show plots
plt.tight_layout()
plt.show()

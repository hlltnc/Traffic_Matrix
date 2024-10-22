import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_state_city
from qiskit_aer.noise import NoiseModel, pauli_error

# Create a circuit that generates 2 Bell states (4 qubits total)
def create_bell_state_circuit():
    qc = QuantumCircuit(4, 4)
    # First Bell pair (qubits 0 and 1)
    qc.h(0)
    qc.cx(0, 1)
    # Second Bell pair (qubits 2 and 3)
    qc.h(2)
    qc.cx(2, 3)
    return qc

# Generate circuits for measurements in X, Y, Z bases
def generate_tomography_circuits():
    qc = create_bell_state_circuit()
    circuits = []

    # Base Z: Standard measurement in the computational basis
    qc_z = qc.copy()
    qc_z.measure_all()
    circuits.append(qc_z)

    # Base X: Apply Hadamard before measurement
    qc_x = qc.copy()
    qc_x.h(0)
    qc_x.h(1)
    qc_x.h(2)
    qc_x.h(3)
    qc_x.measure_all()
    circuits.append(qc_x)

    # Base Y: Apply S-dagger and Hadamard before measurement
    qc_y = qc.copy()
    qc_y.sdg(0)
    qc_y.h(0)
    qc_y.sdg(1)
    qc_y.h(1)
    qc_y.sdg(2)
    qc_y.h(2)
    qc_y.sdg(3)
    qc_y.h(3)
    qc_y.measure_all()
    circuits.append(qc_y)

    return circuits

# Add depolarizing noise to the circuit
# def add_depolarizing_noise():
#     noise_model = NoiseModel()
#     error_1q = depolarizing_error(0.01, 1)  # 1% error rate for single-qubit gates
#     error_2q = depolarizing_error(0.02, 2)  # 2% error rate for two-qubit gates
#     noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
#     noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
#     return noise_model


# Function to simulate bit-flip noise
def add_bit_flip_noise():
    noise_model = NoiseModel()
    error_1q = pauli_error([('X', 0.05), ('I', 0.95)])  # 5% chance of bit-flip, 95% no error
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x'])
    error_2q = pauli_error([('XX', 0.05), ('II', 0.95)])  # 5% chance of bit-flip on both qubits
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model
# Perform tomography simulation with depolarizing noise
def perform_tomography(circuits, shots=1024):
    simulator = Aer.get_backend('qasm_simulator')
    noise_model = add_bit_flip_noise()  # Add depolarizing noise

    # Execute the circuits (without names)
    result = execute(circuits, simulator, noise_model=noise_model, shots=shots).result()

    # Retrieve counts based on index
    counts = [result.get_counts(i) for i in range(len(circuits))]
    
    return counts

# Manually reconstruct the quantum state using measurements from X, Y, Z bases
def reconstruct_density_matrix(counts, shots):
    # Assume 4 qubits, so 16x16 density matrix
    num_qubits = 4
    dim = 2 ** num_qubits

    # Initialize an empty density matrix
    rho = np.zeros((dim, dim), dtype=complex)

    # Measurement operators for Z, X, Y bases
    measurement_ops = {
        '0': np.array([[1, 0], [0, 0]]),  # |0⟩⟨0|
        '1': np.array([[0, 0], [0, 1]])   # |1⟩⟨1|
    }

    # Handle measurements from all bases (Z, X, Y)
    for basis_idx, basis_counts in enumerate(counts):
        for outcome, count in basis_counts.items():
            prob = count / shots

            # Extract the first 4 bits of the outcome string (the qubits)
            outcome = outcome.split()[0]

            # Process each outcome string, treat each bit as its own measurement outcome
            ops = [measurement_ops[bit] for bit in outcome]

            # Calculate Kronecker product of the operators
            operator_kron = ops[0]
            for op in ops[1:]:
                operator_kron = np.kron(operator_kron, op)

            # Update the density matrix
            rho += prob * operator_kron

    # Ensure the trace is not zero
    trace_rho = np.trace(rho)
    if np.isclose(trace_rho, 0):
        raise ValueError("Trace of the density matrix is zero, invalid state.")

    # Normalize the matrix by the trace
    rho /= trace_rho

    # Symmetrize the matrix to enforce Hermiticity
    rho = (rho + rho.conjugate().T) / 2

    # Ensure the matrix is positive semi-definite (non-negative eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)  # Force non-negative eigenvalues
    rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conjugate().T

    return rho

# Validate that the density matrix is a valid quantum state
def validate_density_matrix(rho):
    # Check if the matrix is Hermitian
    if not np.allclose(rho, rho.conjugate().T):
        raise ValueError("Density matrix is not Hermitian")

    # Check if the matrix is positive semi-definite (all eigenvalues should be >= 0)
    eigenvalues = np.linalg.eigvals(rho)
    if np.any(eigenvalues < 0):
        raise ValueError("Density matrix has negative eigenvalues")

    # Check if the trace of the matrix is 1
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Density matrix trace is not equal to 1")

# Main Execution
circuits = generate_tomography_circuits()
counts = perform_tomography(circuits, shots=1024)

# Debug: Print counts for each basis
for i, basis in enumerate(['Z', 'X', 'Y']):
    print(f"Counts for {basis}-basis:")
    print(counts[i])

# Reconstruct the quantum state
rho_reconstructed = reconstruct_density_matrix(counts, shots=1024)

# Validate the reconstructed density matrix
try:
    validate_density_matrix(rho_reconstructed)
    print("Density matrix is valid.")
except ValueError as e:
    print(f"Density matrix validation failed: {e}")

# Ideal Bell state density matrix
bell_state = create_bell_state_circuit()
ideal_density_matrix = DensityMatrix.from_instruction(bell_state)

# Convert reconstructed density matrix to a valid DensityMatrix object
rho_reconstructed_dm = DensityMatrix(rho_reconstructed)

# Calculate the fidelity between the reconstructed state and the ideal Bell state
fidelity = state_fidelity(ideal_density_matrix, rho_reconstructed_dm)
##print(f"Fidelity of the reconstructed state: {fidelity}")
##print(rho_reconstructed)
# Visualize the reconstructed density matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer.noise import NoiseModel, pauli_error

# (rest of the code remains unchanged)

# Function to create a 3D bar plot of the real or imaginary parts of the density matrix with better readable binary labels
def plot_3d_density_matrix(rho, part='real', save_as_pdf=False, filename='density_matrix.pdf'):
    if part == 'real':
        matrix_part = np.real(rho)
        title = "Real Part of Density Matrix"
    elif part == 'imaginary':
        matrix_part = np.imag(rho)
        title = "Imaginary Part of Density Matrix"
    else:
        raise ValueError("Invalid part specified. Use 'real' or 'imaginary'.")

    fig = plt.figure(figsize=(16, 14))  # Increased figure size
    ax = fig.add_subplot(111, projection='3d')

    # Set up the plot grid and bar positions
    num_rows, num_cols = matrix_part.shape
    xpos, ypos = np.meshgrid(range(num_rows), range(num_cols), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    # Heights of the bars
    dz = matrix_part.flatten()

    # Bar plot
    ax.bar3d(xpos, ypos, zpos, dx=0.5, dy=0.5, dz=dz, shade=True)

    # Setting labels with binary tick marks
    #ax.set_xlabel('Row Index', labelpad=20)  # Reduced labelpad to bring labels closer
    #ax.set_ylabel('Column Index', labelpad=20)  # Reduced labelpad to bring labels closer
    ax.set_zlabel('Value', labelpad=2)
    #ax.set_title(title, pad=30)

    # Convert tick labels to binary format
    row_labels = [format(i, f'0{int(np.log2(num_rows))}b') for i in range(num_rows)]
    col_labels = [format(i, f'0{int(np.log2(num_cols))}b') for i in range(num_cols)]

    # Set ticks with binary labels, and adjust spacing for better readability
    ax.set_xticks(range(num_rows))
    ax.set_xticklabels(row_labels, rotation=45, ha='right', fontsize=10, weight='bold')  # Adjusted rotation and size
    ax.set_yticks(range(num_cols))
    ax.set_yticklabels(col_labels, rotation=-30, ha='left', fontsize=10, weight='bold')  # Changed angle and alignment

    # Adjust tick positions to make them more spaced out or closer
    ax.tick_params(axis='x', pad=5)  # Reduced padding to bring x-axis labels closer
    ax.tick_params(axis='y', pad=5)  # Reduced padding to bring y-axis labels closer

    # Show the plot
    plt.show()

    # Save as PDF if specified
    if save_as_pdf:
        fig.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Plot saved as {filename}")

# Main Execution
circuits = generate_tomography_circuits()
counts = perform_tomography(circuits, shots=1024)

# Reconstruct the quantum state
rho_reconstructed = reconstruct_density_matrix(counts, shots=1024)

# Validate the reconstructed density matrix
try:
    validate_density_matrix(rho_reconstructed)
    print("Density matrix is valid.")
except ValueError as e:
    print(f"Density matrix validation failed: {e}")

# Plot real part and save as PDF
plot_3d_density_matrix(rho_reconstructed, part='real', save_as_pdf=True, filename='real_density_matrix.pdf')

# Plot imaginary part and save as PDF
plot_3d_density_matrix(rho_reconstructed, part='imaginary', save_as_pdf=True, filename='imaginary_density_matrix.pdf')

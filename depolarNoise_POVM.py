from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer  # Aer simülatörünü kullanacağız
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel, depolarizing_error
from numpy.random import default_rng
from povm_toolbox.sampler import POVMSampler
from qiskit.primitives import StatevectorSampler as Sampler
from povm_toolbox.library import ClassicalShadows


# 1. Alice'in dolanık qubit oluşturması (Bell durumu)
qc = QuantumCircuit(2)  # 2 qubit için devre

qc.h(0)  # İlk qubit'e Hadamard kapısı uygula
qc.cx(0, 1)  # İlk qubit ile ikinci qubit arasında CNOT kapısı uygula (dolanıklık oluştur)

# 2. Depolarizasyon gürültüsü oluştur
p = 0.1  # Depolarizasyon olasılığı
depolarizing_noise = depolarizing_error(p, 1)  # İlk parametre olasılık, ikinci parametre qubit sayısı

# Gürültü modelini tanımla ve Bob'a giden qubit'e uygula
noise_model_d = NoiseModel()
noise_model_d.add_quantum_error(depolarizing_noise, ['id', 'u3'], [1])  # İkinci qubit (Bob'a giden qubit)

# 3. Simülasyonu başlat (Gürültü modeli ile)
simulator = Aer.get_backend('qasm_simulator')



qc.measure_all()  # Alice ve Bob'un ölçüm yapması için ölçüm ekleniyor

# 4. Devreyi transpile et ve çalıştır
qc_compiled = transpile(qc, simulator)
job = simulator.run(qc_compiled, noise_model=noise_model_d, shots=1024)  # Fazladan simulator argümanı kaldırıldı

result = job.result()

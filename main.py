import numpy as np
from qiskit import QuantumCircuit
from scripts.qrc_model import run_qrc_model

# Build Quantum Circuit
def build_quantum_circuit(n_qubits, depth):
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        for qubit in range(n_qubits):
            qc.rx(np.random.uniform(0, 2 * np.pi), qubit)
            qc.rz(np.random.uniform(0, 2 * np.pi), qubit)
        for qubit in range(n_qubits - 1):
            qc.cz(qubit, qubit + 1)
    return qc

# Visualize Quantum Circuit
def visualize_quantum_circuit(qc):
    qc.draw('mpl')

if __name__ == "__main__":
    # Build and visualize the quantum circuit
    n_qubits = 4
    depth = 3
    qc = build_quantum_circuit(n_qubits, depth)
    visualize_quantum_circuit(qc)

    # Run the QRC model
    recall = run_qrc_model('dataset/german.data-numeric')
    print(f"Recall Score: {recall}")

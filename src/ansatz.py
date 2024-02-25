from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal


class Ansatz:

    def __init__(self, circuit: QuantumCircuit | TwoLocal, entanglement: str | None = None, cnot_count: int | None = None, depth: int | None = None, param_count: int | None = None) -> None:
        if isinstance(circuit, TwoLocal):
            self.circuit = circuit.decompose()
            self.entanglement = circuit.entanglement
            self.cnot_count = self.circuit.count_ops()['cx']
            self.depth = self.circuit.depth()
            self.param_count = self.circuit.num_parameters
        else:
            self.circuit = circuit
            self.entanglement = entanglement
            self.cnot_count = cnot_count
            self.depth = depth
            self.param_count = param_count

    def draw(self):
        return self.circuit.draw('mpl')


def get_basic_ansatzes() -> List[Ansatz]:
    NUM_QUBITS = 4
    INSERT_BARRIES = True
    ROTATION_BLOCKS = ['ry']
    ENTANGLEMENT_BLOCKS = ['cx']
    REPS = 3

    return [
        Ansatz(circuit=TwoLocal(
            num_qubits=NUM_QUBITS,
            rotation_blocks=ROTATION_BLOCKS,
            entanglement_blocks=ENTANGLEMENT_BLOCKS,
            entanglement='linear',
            reps=REPS,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=NUM_QUBITS,
            rotation_blocks=ROTATION_BLOCKS,
            entanglement_blocks=ENTANGLEMENT_BLOCKS,
            entanglement='full',
            reps=REPS,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=NUM_QUBITS,
            rotation_blocks=ROTATION_BLOCKS,
            entanglement_blocks=ENTANGLEMENT_BLOCKS,
            entanglement='circular',
            reps=REPS,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=NUM_QUBITS,
            rotation_blocks=ROTATION_BLOCKS,
            entanglement_blocks=ENTANGLEMENT_BLOCKS,
            entanglement='pairwise',
            reps=REPS,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=NUM_QUBITS,
            rotation_blocks=ROTATION_BLOCKS,
            entanglement_blocks=ENTANGLEMENT_BLOCKS,
            entanglement='sca',
            reps=REPS,
            insert_barriers=INSERT_BARRIES,
        ))
    ]
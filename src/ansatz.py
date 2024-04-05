from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal


class Ansatz:

    def __init__(self, circuit: QuantumCircuit | TwoLocal, entanglement: str = None, cnot_count: int = None, depth: int = None, param_count: int = None, num_layers: int = None) -> None:
        if isinstance(circuit, TwoLocal):
            self.circuit = circuit.decompose()
            self.entanglement = circuit.entanglement
            self.cnot_count = self.circuit.count_ops()['cx']
            self.depth = self.circuit.depth()
            self.param_count = self.circuit.num_parameters
            self.num_layers = circuit.reps
        else:
            self.circuit = circuit
            self.entanglement = entanglement
            self.cnot_count = cnot_count
            self.depth = depth
            self.param_count = param_count
            self.num_layers = num_layers

    def draw(self):
        return self.circuit.draw('mpl')


def get_basic_ansatzes(num_qubits: int = 4, rotation_blocks: List[str] = ['ry'], entanglement_blocks: List[str] = ['cx'], num_layers: int = 3) -> List[Ansatz]:

    INSERT_BARRIES = True

    return [
        Ansatz(circuit=TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement='linear',
            reps=num_layers,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement='reverse_linear',
            reps=num_layers,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement='full',
            reps=num_layers,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement='circular',
            reps=num_layers,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement='pairwise',
            reps=num_layers,
            insert_barriers=INSERT_BARRIES,
        )), Ansatz(circuit=TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement='sca',
            reps=num_layers,
            insert_barriers=INSERT_BARRIES,
        ))
    ]

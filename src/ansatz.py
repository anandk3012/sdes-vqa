from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, EfficientSU2

def ansatz_cy_1(num_qubits: int):
    """
    Build a CY-based ansatz:
    - Apply H on all qubits
    - Apply RY(θ_i) on each qubit
    - Apply a ring of CY gates
    Returns:
        qc: QuantumCircuit with symbolic parameters
        params: list of ParameterVector symbols [θ_0, θ_1, ..., θ_{num_qubits-1}]
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector('θ', length=num_qubits)

    # Hadamards
    qc.h(range(num_qubits))

    # RY rotations
    for i in range(num_qubits):
        qc.ry(params[i], i)

    # Ring of CY entanglers
    for i in range(num_qubits):
        qc.cy(i, (i + 1) % num_qubits)

    return qc, list(params)


def layered_ansatz(num_qubits: int, layers: int = 2):
    """
    Build a layered CY ansatz:
    - Apply H on all qubits
    - For each layer:
        • RY(θ_{l * num_qubits + i}) on qubit i
        • CY(i, (i+1)%num_qubits) for all i
    Args:
        num_qubits: number of qubits
        layers: number of repeated RY+CY layers
    Returns:
        qc: QuantumCircuit with symbolic parameters
        params: list of ParameterVector symbols of length num_qubits * layers
    """
    total_params = num_qubits * layers
    params = ParameterVector('θ', length=total_params)
    qc = QuantumCircuit(num_qubits)

    # Initial Hadamards
    qc.h(range(num_qubits))

    # Layered structure
    for l in range(layers):
        for i in range(num_qubits):
            qc.ry(params[l * num_qubits + i], i)
        for i in range(num_qubits):
            qc.cy(i, (i + 1) % num_qubits)

    return qc, list(params)


def create_improved_ansatz(num_qubits: int = 10, reps: int = 5):
    """
    Build a better layered “U + Entanglement” ansatz:
    - Apply H on all qubits
    - Repeat for 'reps' times:
        • U(θ_i, φ_i, λ_i) on each qubit
        • Entangle qubits with staggered CX pattern depending on layer
    Args:
        num_qubits: Number of qubits
        reps: Number of layers (depth) to stack
    
    Returns:
        qc: QuantumCircuit with symbolic parameters
        theta_list: list of θ Parameters
        phi_list: list of φ Parameters
        lambda_list: list of λ Parameters
    """
    θ = ParameterVector('θ', num_qubits * reps)
    φ = ParameterVector('φ', num_qubits * reps)
    λ = ParameterVector('λ', num_qubits * reps)
    qc = QuantumCircuit(num_qubits)

    # Initial Hadamards
    qc.h(range(num_qubits))

    # For each layer
    for rep in range(reps):
        offset = rep * num_qubits
        # Parameterized single-qubit rotations
        for i in range(num_qubits):
            qc.u(θ[offset + i], φ[offset + i], λ[offset + i], i)

        # Layer-dependent entanglement pattern
        if rep % 2 == 0:
            # Even layers: standard neighbor entanglement
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)
        else:
            # Odd layers: skip neighbor entanglement
            for i in range(num_qubits):
                qc.cx(i, (i + 2) % num_qubits)

    return qc, list(θ), list(φ), list(λ)


def create_zzfeature_map(num_qubits: int = 10, reps: int = 2):
    """
    Build a ZZFeatureMap followed by a TwoLocal variation form:
    - feature map: ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement='linear')
    - var form: TwoLocal(num_qubits, ['ry','rz'], 'cz', reps=reps, entanglement='linear')
    Returns:
        qc: QuantumCircuit representing feature_map ∘ var_form
    """
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement='linear')
    var_form = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=reps, entanglement='linear')
    vqc = feature_map.compose(var_form)
    return vqc
# def create_improved_ansatz(num_qubits: int = 10, reps: int = 2):
#     """
#     Build a ZZFeatureMap followed by a TwoLocal variation form:
#     - feature map: ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement='linear')
#     - var form: TwoLocal(num_qubits, ['ry','rz'], 'cz', reps=reps, entanglement='linear')
#     Returns:
#         qc: QuantumCircuit representing feature_map ∘ var_form
#     """
#     feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement='linear')
#     var_form = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=reps, entanglement='linear')
#     vqc = feature_map.compose(var_form)
#     return vqc


def create_2local_ansatz(num_qubits: int = 10, reps: int = 3):
    """
    Build a TwoLocal ansatz:
    - Rotation blocks: ['ry','rz']
    - Entanglement blocks: 'cz'
    - Entanglement pattern: 'linear'
    - reps: 1
    - insert_barriers: True
    Returns:
        qc: TwoLocal quantum circuit
    """
    qc = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cz',
        entanglement='linear',
        reps=1,
        insert_barriers=True
    )
    return qc
# def create_improved_ansatz(num_qubits: int = 10, reps: int = 3 ):
#     """
#     Build a TwoLocal ansatz:
#     - Rotation blocks: ['ry','rz']
#     - Entanglement blocks: 'cz'
#     - Entanglement pattern: 'linear'
#     - reps: 1
#     - insert_barriers: True
#     Returns:
#         qc: TwoLocal quantum circuit
#     """
#     qc = TwoLocal(
#         num_qubits=num_qubits,
#         rotation_blocks=['ry', 'rz'],
#         entanglement_blocks='cz',
#         entanglement='linear',
#         reps=1,
#         insert_barriers=True
#     )
#     return qc


def create_efficient_su2_ansatz(num_qubits: int = 10, reps: int = 4):

# def create_improved_ansatz(num_qubits: int = 10, reps: int = 4):
    """
    Build an EfficientSU2 ansatz:
    - reps: 1
    - entanglement: 'full'
    - insert_barriers: True
    Returns:
        qc: EfficientSU2 quantum circuit
    """
    qc = EfficientSU2(
        num_qubits=num_qubits,
        reps=reps,
        entanglement='full',
        insert_barriers=True
    )
    return qc

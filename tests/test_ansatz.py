# import pytest
# from qiskit.circuit import ParameterVector
# from src.ansatz import (
#     ansatz_cy_1,
#     layered_ansatz,
#     create_improved_ansatz,
#     create_zzfeature_map,
#     create_2local_ansatz,
#     create_efficient_su2_ansatz,
# )


# def test_ansatz_cy_1_dimensions():
#     num_qubits = 5
#     qc, params = ansatz_cy_1(num_qubits)
#     # Expect num_qubits parameters, one RY per qubit
#     assert len(params) == num_qubits
#     assert isinstance(params[0], ParameterVector) or hasattr(params[0], "name")
#     # Circuit should have at least num_qubits Hadamards and num_qubits RYs
#     assert qc.num_qubits == num_qubits


# def test_layered_ansatz_parameter_count():
#     num_qubits = 4
#     layers = 3
#     qc, params = layered_ansatz(num_qubits, layers=layers)
#     # Expect num_qubits * layers parameters
#     assert len(params) == num_qubits * layers
#     assert qc.num_qubits == num_qubits


# def test_create_improved_ansatz_structure():
#     num_qubits = 6
#     reps = 5
#     qc, theta_list, phi_list, lambda_list = create_improved_ansatz(num_qubits=num_qubits, reps=reps)
#     # Each list should have length num_qubits
#     assert len(theta_list) == num_qubits * reps
#     assert len(phi_list) == num_qubits * reps
#     assert len(lambda_list) == num_qubits * reps
#     assert qc.num_qubits == num_qubits


# def test_zzfeature_map_and_variational_compose():
#     num_qubits = 3
#     vqc = create_zzfeature_map(num_qubits=num_qubits, reps=1)
#     # Expect a QuantumCircuit with num_qubits qubits
#     assert vqc.num_qubits == num_qubits


# def test_2local_and_efficient_su2_qubits():
#     num_qubits = 7
#     qc2 = create_2local_ansatz(num_qubits=num_qubits)
#     qcesu2 = create_efficient_su2_ansatz(num_qubits=num_qubits)
#     assert qc2.num_qubits == num_qubits
#     assert qcesu2.num_qubits == num_qubits

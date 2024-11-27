# Copyright (C) 2024 Matthias Deiml, Daniel Peterseim - All rights reserved

import qiskit as qk
import numpy as np
import qiskit_aer
from qiskit.circuit.library import ZGate
from qiskit.primitives import BaseEstimatorV2 as Estimator, StatevectorEstimator


# The code for the Newton's method is very similar to the fixed-point iteration.
# The three main differences are:
#
# * The implementation of the Jacobian Dg in addition to g (see
#   `block_encoding_Dg`).
#
# * The implementation of a linear solver (see `invert`).
#
# * Ancilla bits have to be handled more carefully due to the use of QSVT in the
#   linear solver

backend = qiskit_aer.Aer.get_backend("aer_simulator")
estimator = StatevectorEstimator()


class BlockEncoding:

    def __init__(self, U: qk.QuantumCircuit, cU: qk.QuantumCircuit, norm: float, size: int=1):
        self.U = U        # non-controlled version of unitary of block encoding
        self.cU = cU      # controlled version of unitary of block encoding
        self.norm = norm  # normalization of block encoding
        self.size = size  # number of non-ancilla qubits


def block_encoding_g(x: BlockEncoding, target_norm: float | None = None) -> BlockEncoding:
    norm = np.sqrt(2) + (2/8) * x.norm ** 2

    if target_norm is not None:
        assert target_norm > norm
        diff = target_norm - norm
        angle = 2 * np.arctan(np.sqrt((np.sqrt(2) + diff) / ((2/8) * x.norm ** 2)))
        angle2 = 2 * np.arccos(np.sqrt(2) / (np.sqrt(2) + diff))
        norm = target_norm
    else:
        angle = 2 * np.arctan(np.sqrt(np.sqrt(2) / ((2/8) * x.norm ** 2)))

    c = qk.QuantumRegister(1, name="c")
    x1 = qk.QuantumRegister(1, name="x1")
    x2 = qk.QuantumRegister(1, name="x2")
    add = qk.QuantumRegister(1, name="add")
    a1 = qk.QuantumRegister(x.size - 1, name="a1")
    a2 = qk.QuantumRegister(1, name="a2")
    a3 = qk.QuantumRegister(x.cU.num_qubits - x.size - 1, name="a3")

    if x.size >= 2:
        U = qk.QuantumCircuit(x1, x2, add, a1, a2, a3, name="U_g")
    else:
        U = qk.QuantumCircuit(x1, x2, add, a1, a3, name="U_g")
    U.ry(angle, add)
    U.ch(add, x1)

    if target_norm is not None:
        U.cry(angle2, add, x2)

    U.x(add)
    U.z(add)
    U.append(x.cU, [add, x1[0]] + a1[:] + a3[:])
    if x.size >= 2:
        U.mcx(add[:] + a1[:], a2[:], ctrl_state=1)
        U.cx(add[:], a2)
    U.append(x.cU, [add, x2[0]] + a1[:] + a3[:])
    U.ch(add, x1)
    U.ch(add, x2)
    U.ccx(add, x1, x2)
    U.ry(-(np.pi - angle), add)

    a4 = qk.QuantumRegister(1, name="a4")

    if x.size >= 2:
        cU = qk.QuantumCircuit(c, x1, x2, add, a1, a2, a3, a4, name="cU_g")
    else:
        cU = qk.QuantumCircuit(c, x1, x2, add, a1, a3, a4, name="cU_g")
    cU.ry(angle, add)
    cU.ccx(c, add, a4)
    cU.ch(a4, x1)
    if target_norm is not None:
        cU.cry(angle2, a4, x2)
    cU.cx(c, a4)
    cU.z(a4)
    cU.append(x.cU, [a4[0], x1[0]] + a1[:] + a3[:])
    if x.size >= 2:
        cU.mcx(a4[:] + a1[:], a2[:], ctrl_state=1)
        cU.cx(a4, a2)
    cU.append(x.cU, [a4[0], x2[0]] + a1[:] + a3[:])
    cU.ch(a4, x1)
    cU.ch(a4, x2)
    cU.ccx(a4, x1, x2)
    cU.cx(c, a4)
    cU.ccx(c, add, a4)
    cU.ry(-angle, add)

    if x.size >= 2:
        size = 3 + x.size
    else:
        size = 2 + x.size

    return BlockEncoding(U, cU, norm, size)


def amplify(inp: BlockEncoding, k: int) -> BlockEncoding:

    assert k % 2 == 1

    c = qk.QuantumRegister(1, name="c")
    x = qk.QuantumRegister(1, name="x")
    a1 = qk.QuantumRegister(inp.size-1, name="a1")
    a2 = qk.QuantumRegister(inp.U.num_qubits-inp.size, name="a2")

    U = qk.QuantumCircuit(x, a1, a2)
    for i in range(k):
        if i % 2 == 0:
            U.append(inp.U, x[:] + a1[:] + a2[:])
        else:
            U.append(inp.U.inverse(), x[:] + a1[:] + a2[:])

        if False and i == (k - 1):
            if ((k - 1) / 2) % 2 == 1:
                U.append(qk.circuit.library.GlobalPhaseGate(np.pi))
        elif i % 2 == 1:
            U.x(x)
            U.append(ZGate().control(inp.size-1, ctrl_state=0), a1[:] + x[:])
            U.x(x)
        else:
            U.x(a1[-1])
            U.append(ZGate().control(inp.size-2, ctrl_state=0), a1[:])
            U.x(a1[-1])

    a2 = qk.QuantumRegister(max(inp.U.num_qubits-inp.size, inp.cU.num_qubits - inp.size - 1), name="a2")

    cU = qk.QuantumCircuit(c, x, a1, a2)
    for i in range(k):
        if i % 2 == 0:
            if i == k - 1:
                cU.append(inp.cU, c[:] + x[:] + a1[:] + a2[:inp.cU.num_qubits - inp.size - 1])
            else:
                cU.append(inp.U, x[:] + a1[:] + a2[:inp.U.num_qubits - inp.size])
        else:
            cU.append(inp.U.inverse(), x[:] + a1[:] + a2[:inp.U.num_qubits - inp.size])
        if False and i == (k - 1):
            if ((k - 1) / 2) % 2 == 1:
                cU.z(c)
        elif i % 2 == 1:
            cU.append(ZGate().control(inp.size, ctrl_state=0), x[:] + a1[:] + c[:])
        else:
            cU.append(ZGate().control(inp.size-1, ctrl_state=0), a1[:] + c[:])

    print(U)
    print(cU)

    return BlockEncoding(U, cU, inp.norm * np.sin(np.pi/(2 * k)), inp.size)


# This implements a block encoding of Dg the negative(!) Jacobian -Dg(x) given a
# block encoding of x.
def block_encoding_Dg(x: BlockEncoding) -> BlockEncoding:
    norm = (2/4) * x.norm

    c = qk.QuantumRegister(1, name="c")
    x1 = qk.QuantumRegister(1, name="x1")
    x2 = qk.QuantumRegister(1, name="x2")
    a1 = qk.QuantumRegister(x.size - 1, name="a1")
    a2 = qk.QuantumRegister(x.U.num_qubits - x.size, name="a2")

    # The quadratic part is implemented as in `block_encoding_g`, the contant
    # part is dropped.
    U = qk.QuantumCircuit(x1, x2, a1, a2, name="U_Dg")
    U.append(x.U, [x2[0]] + a1[:] + a2[:])
    U.h(x1)
    U.h(x2)
    U.cx(x1, x2)

    a2 = qk.QuantumRegister(x.cU.num_qubits - x.size - 1, name="a2")

    cU = qk.QuantumCircuit(c, x1, x2, a1, a2, name="cU_Dg")
    cU.append(x.cU, [c, x2[0]] + a1[:] + a2[:])
    cU.ch(c, x1)
    cU.ch(c, x2)
    cU.ccx(c, x1, x2)

    return BlockEncoding(U, cU, norm, x.size + 1)


# This implements the linear solver based on QSVT.
# [1] https://arxiv.org/abs/1806.01838
# [2] https://arxiv.org/abs/2105.02859
def inverse(Dg: BlockEncoding) -> BlockEncoding:
    # Angles computed using pyqsp [2]
    angles = np.array([-0.003860431530918084, 0.011746088645406612, -0.04164716560407932, 0.11008518322901666, -0.43656650933198426, -0.007689861025225564, -0.5370132056833176, 0.8931764280015879, -1.320709604226413, 1.8208830504090836, -2.2484162224410738, -0.5370131995188709, -0.0076898550452888514, 2.705026142033692, 0.11008518151197777, -0.041647166831564, -3.1298465650961154, 1.5669358952445285])

    # Turn Wx convention angles to R convention angles [1, Corollary 8]
    d = len(angles)-1
    angles_R = np.zeros(d)
    angles_R[0] = angles[0] + angles[-1] + (d-1) * np.pi/2
    angles_R[1:] = angles[1:d] - np.pi/2
    angles = angles_R
    qsp_norm = 2.17970512

    assert len(angles) % 2 == 1

    c = qk.QuantumRegister(1, name="c")
    x1 = qk.QuantumRegister(1, name="x1")
    b = qk.QuantumRegister(1, name="b")
    a1 = qk.QuantumRegister(Dg.size-1, name="a1")
    a2 = qk.QuantumRegister(Dg.U.num_qubits - Dg.size, name="a2")

    # The block encoding is just the usual QSVT [1, Figure 1]
    U = qk.QuantumCircuit(x1, b, a1, a2, name="U_Dg_inv")
    U.h(b)

    for (i, angle) in enumerate(reversed(angles)):
        if i % 2 == 0:
            U.append(Dg.U.inverse(), x1[:] + a1[:] + a2[:])
        else:
            U.append(Dg.U, x1[:] + a1[:] + a2[:])

        U.mcx(a1, b, ctrl_state=0)
        U.rz(-2 * angle, b)
        U.mcx(a1, b, ctrl_state=0)

    U.h(b)

    a2 = qk.QuantumRegister(max(
        Dg.cU.num_qubits - 1 - Dg.size,
        Dg.U.num_qubits - Dg.size,
    ), name="a2")

    cU = qk.QuantumCircuit(c, x1, b, a1, a2, name="U_Dg_inv")
    cU.h(b)

    for (i, angle) in enumerate(reversed(angles)):
        if i % 2 == 0:
            if i == 0:
                cU.append(Dg.cU.inverse(), c[:] + x1[:] + a1[:] + a2[:Dg.cU.num_qubits-Dg.size-1])
            else:
                cU.append(Dg.U.inverse(), x1[:] + a1[:] + a2[:Dg.U.num_qubits-Dg.size])
        else:
            cU.append(Dg.U, x1[:] + a1[:] + a2[:Dg.U.num_qubits-Dg.size])

        cU.mcx(a1, b, ctrl_state=0)
        cU.crz(-2 * angle, c, b)
        cU.mcx(a1, b, ctrl_state=0)

    cU.h(b)

    return BlockEncoding(U, cU, qsp_norm / Dg.norm, Dg.size+1)


def estimate_ie(be: BlockEncoding, estimator: Estimator) -> float:
    print(f"estimating circuit with {be.U.num_qubits} qubits")
    zero = qk.quantum_info.SparsePauliOp(["I", "Z"], coeffs=[0.5, 0.5])
    assert be.size >= 2
    zeros = zero
    for _ in range(be.size-2):
        zeros = zeros ^ zero

    obs = qk.quantum_info.SparsePauliOp("I" * (be.U.num_qubits - be.size)) ^ zeros ^ qk.quantum_info.SparsePauliOp("I")
    circ = be.U
    return np.sqrt(estimator.run([(circ, obs)]).result()[0].data.evs)


# Impelements the multiplication and addition Ay + b where for the Newton's
# method we set A = Dg(x)^{-1}, y = g(x), b = x.
def axpy(Dg_inv: BlockEncoding, gx: BlockEncoding, x: BlockEncoding) -> BlockEncoding:
    norm = x.norm + Dg_inv.norm * gx.norm
    angle = 2 * np.arctan(np.sqrt(x.norm / (Dg_inv.norm * gx.norm)))

    assert Dg_inv.size > 1
    assert gx.size > 1

    c = qk.QuantumRegister(1, name="c")
    x1 = qk.QuantumRegister(1, name="x1")
    add = qk.QuantumRegister(1, name="add")
    a1 = qk.QuantumRegister(max(Dg_inv.size - 1, gx.size - 1, x.size - 1), name="a1")
    a2 = qk.QuantumRegister(1)
    a3 = qk.QuantumRegister(max(Dg_inv.cU.num_qubits - Dg_inv.size - 1, gx.cU.num_qubits - gx.size - 1, x.cU.num_qubits - x.size - 1), name="a2")

    U = qk.QuantumCircuit(x1, add, a1, a2, a3, name="U_Dg_inv_g")
    U.ry(angle, add)
    U.append(x.cU, add[:] + x1[:] + a1[:x.size-1] + a3[:x.cU.num_qubits-x.size-1])

    U.x(add)

    U.append(gx.cU, add[:] + x1[:] + a1[:gx.size-1] + a3[:gx.cU.num_qubits-gx.size-1])
    U.mcx(add[:] + a1[:min(gx.size - 1, Dg_inv.size - 1)], a2, ctrl_state=1)
    U.cx(add[:], a2)
    U.append(Dg_inv.cU, add[:] + x1[:] + a1[:Dg_inv.size - 1] + a3[:Dg_inv.cU.num_qubits-Dg_inv.size-1])

    U.ry(-(np.pi - angle), add)

    a4 = qk.QuantumRegister(1, name="a4")

    cU = qk.QuantumCircuit(c, x1, add, a1, a2, a3, a4, name="cU_Dg_inv_g")
    cU.ry(angle, add)
    cU.ccx(c, add, a4)
    cU.append(x.cU, a4[:] + x1[:] + a1[:x.size-1] + a3[:x.cU.num_qubits-x.size-1])

    cU.cx(c, a4)

    cU.append(gx.cU, a4[:] + x1[:] + a1[:gx.size-1] + a3[:gx.cU.num_qubits-gx.size-1])
    cU.mcx(a4[:] + a1[:min(gx.size - 1, Dg_inv.size - 1)], a2, ctrl_state=1)
    cU.cx(a4, a2)
    cU.append(Dg_inv.cU, a4[:] + x1[:] + a1[:Dg_inv.size - 1] + a3[:Dg_inv.cU.num_qubits-Dg_inv.size-1])

    cU.cx(c, a4)
    cU.ccx(c, add, a4)

    cU.ry(-angle, add)

    return BlockEncoding(U, cU, norm, len(a1) + 3)


# Finally we can define the amplified encoding of a single newton step.
def newton_step(x: BlockEncoding, estimator: Estimator):
    gx = block_encoding_g(x)
    Dg_inv = inverse(block_encoding_Dg(x))
    next = axpy(Dg_inv, gx, x)

    # To get perfect information efficiency, we intentionally decrease the
    # information efficiency of g(x).
    ie = estimate_ie(next, estimator)
    k = 2 * int(np.ceil(0.25 * (np.pi / np.arcsin(ie) - 2))) + 1
    add_subnorm = np.sin(np.pi/(2 * k)) / ie
    target_norm = next.norm / add_subnorm
    target_norm_gx = (target_norm - x.norm) / Dg_inv.norm

    next_sub = axpy(Dg_inv, block_encoding_g(x, target_norm_gx), x)
    return amplify(next_sub, k)


# Classical reference implementation of g and its Jacobian

def g(x):
    Hx = np.array([[1, 1], [1, -1]]) @ x
    return np.array([1, 1]) - 1/8 * Hx * Hx


def Dg(x):
    return -0.25 * np.array([[x[0]+x[1], x[0]+x[1]], [x[0]-x[1], x[1]-x[0]]])


x_ref = np.array([2, 0.25])
U_x = qk.QuantumCircuit(1, name="U_x0")
U_x.ry(2 * np.arctan(x_ref[1]/x_ref[0]), 0)
cU_x = qk.QuantumCircuit(2, name="cU_x0")
cU_x.cry(2 * np.arctan(x_ref[1]/x_ref[0]), 0, 1)
x0 = BlockEncoding(U_x, cU_x, np.linalg.norm(x_ref))

from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

N = 2
x = x0
for i in range(N):
    x = newton_step(x, estimator)

    obs0 = qk.quantum_info.SparsePauliOp(["I" * x.U.num_qubits, "I" * (x.U.num_qubits - 1) + "Z"], coeffs=[0.5, 0.5])
    circ = x.U
    result = estimator.run([(circ, obs0)]).result()

    p = np.array([result[0].data.evs, 1-result[0].data.evs])
    print(f"x{i+1} (simulated): {np.sqrt(p) * x.norm}")

    x_ref = x_ref - np.linalg.solve(Dg(x_ref), g(x_ref))
    print(f"x{i+1} (reference): {x_ref}")

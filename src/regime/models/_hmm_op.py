"""
regime.models._hmm_op
=====================
Numba-accelerated PyTensor Ops for exact marginalized HMM log-likelihood
and analytical forward-backward gradients.

Kept in a dedicated module (never __main__) so Windows multiprocessing
workers unpickle and resolve `numba_funcify` dispatch cleanly.
"""

from __future__ import annotations

import numba
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch import numba_funcify


@numba.njit(cache=True)
def _hmm_forward_nb(
    log_emit: np.ndarray,
    log_P: np.ndarray,
    log_pi0: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact log-space forward pass returning (loglik, alpha, c)."""
    T, K = log_emit.shape
    alpha = np.empty((T, K), dtype=np.float64)
    c = np.empty(T, dtype=np.float64)

    a0 = log_pi0 + log_emit[0]
    m0 = np.max(a0)
    s0 = np.sum(np.exp(a0 - m0))
    c[0] = m0 + np.log(s0)
    alpha[0] = np.exp(a0 - c[0])

    for t in range(1, T):
        for j in range(K):
            mx = -1e300
            for i in range(K):
                v = np.log(alpha[t - 1, i] + 1e-300) + log_P[i, j]
                if v > mx:
                    mx = v
            sm = 0.0
            for i in range(K):
                sm += np.exp(np.log(alpha[t - 1, i] + 1e-300) + log_P[i, j] - mx)
            alpha[t, j] = mx + np.log(sm) + log_emit[t, j]
        mt = np.max(alpha[t])
        st = np.sum(np.exp(alpha[t] - mt))
        c[t] = mt + np.log(st)
        for j in range(K):
            alpha[t, j] = np.exp(alpha[t, j] - c[t])

    return float(np.sum(c)), alpha, c


@numba.njit(cache=True)
def _hmm_backward_grad_nb(
    log_emit: np.ndarray,
    log_P: np.ndarray,
    log_pi0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact forward-backward pass returning analytical gradients of loglik
    with respect to (log_emit, log_P, log_pi0).
    """
    T, K = log_emit.shape
    _, alpha, c = _hmm_forward_nb(log_emit, log_P, log_pi0)

    log_beta = np.zeros((T, K), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        for i in range(K):
            mx = -1e300
            for j in range(K):
                v = log_P[i, j] + log_emit[t + 1, j] + log_beta[t + 1, j]
                if v > mx:
                    mx = v
            sm = 0.0
            for j in range(K):
                sm += np.exp(log_P[i, j] + log_emit[t + 1, j] + log_beta[t + 1, j] - mx)
            log_beta[t, i] = mx + np.log(sm) - c[t + 1]

    gamma = np.empty((T, K), dtype=np.float64)
    for t in range(T):
        row = np.log(alpha[t] + 1e-300) + log_beta[t]
        rmax = np.max(row)
        e = np.exp(row - rmax)
        gamma[t] = e / np.sum(e)

    xi_sum = np.zeros((K, K), dtype=np.float64)
    for t in range(T - 1):
        mat = np.empty((K, K), dtype=np.float64)
        mx = -1e300
        for i in range(K):
            la = np.log(alpha[t, i] + 1e-300)
            for j in range(K):
                v = la + log_P[i, j] + log_emit[t + 1, j] + log_beta[t + 1, j] - c[t + 1]
                mat[i, j] = v
                if v > mx:
                    mx = v
        sm = 0.0
        for i in range(K):
            for j in range(K):
                mat[i, j] = np.exp(mat[i, j] - mx)
                sm += mat[i, j]
        for i in range(K):
            for j in range(K):
                xi_sum[i, j] += mat[i, j] / sm

    grad_pi0 = gamma[0].copy()
    return gamma, xi_sum, grad_pi0


class HMMGradOp(Op):
    """PyTensor Op computing analytical gradients of HMM marginal log-likelihood."""

    def make_node(self, log_emit, log_P, log_pi0, g_out):
        inputs = [
            pt.as_tensor_variable(log_emit),
            pt.as_tensor_variable(log_P),
            pt.as_tensor_variable(log_pi0),
            pt.as_tensor_variable(g_out),
        ]
        outputs = [inputs[0].type(), inputs[1].type(), inputs[2].type()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        log_emit, log_P, log_pi0, g_out = inputs
        gamma, xi_sum, grad_pi0 = _hmm_backward_grad_nb(
            np.asarray(log_emit, dtype=np.float64),
            np.asarray(log_P, dtype=np.float64),
            np.asarray(log_pi0, dtype=np.float64),
        )
        scale = float(g_out)
        outputs[0][0] = scale * gamma
        outputs[1][0] = scale * xi_sum
        outputs[2][0] = scale * grad_pi0


_hmm_grad_op = HMMGradOp()


class HMMForwardOp(Op):
    """PyTensor Op for exact marginalized Gaussian HMM log-likelihood."""

    def make_node(self, log_emit, log_P, log_pi0):
        inputs = [
            pt.as_tensor_variable(log_emit),
            pt.as_tensor_variable(log_P),
            pt.as_tensor_variable(log_pi0),
        ]
        outputs = [pt.dscalar().type()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        log_emit, log_P, log_pi0 = inputs
        ll, _, _ = _hmm_forward_nb(
            np.asarray(log_emit, dtype=np.float64),
            np.asarray(log_P, dtype=np.float64),
            np.asarray(log_pi0, dtype=np.float64),
        )
        outputs[0][0] = np.array(ll, dtype=np.float64)

    def L_op(self, inputs, outputs, output_grads):
        log_emit, log_P, log_pi0 = inputs
        g_emit, g_P, g_pi0 = _hmm_grad_op(log_emit, log_P, log_pi0, output_grads[0])
        return [g_emit, g_P, g_pi0]


hmm_forward_op = HMMForwardOp()


@numba_funcify.register(HMMForwardOp)
def _numba_funcify_hmm_fwd(op, **kwargs):
    @numba.njit
    def hmm_fwd_impl(log_emit, log_P, log_pi0):
        ll, _, _ = _hmm_forward_nb(log_emit, log_P, log_pi0)
        return np.full((), ll, dtype=np.float64)

    return hmm_fwd_impl


@numba_funcify.register(HMMGradOp)
def _numba_funcify_hmm_grad(op, **kwargs):
    @numba.njit
    def hmm_grad_impl(log_emit, log_P, log_pi0, g_out):
        gamma, xi_sum, grad_pi0 = _hmm_backward_grad_nb(log_emit, log_P, log_pi0)
        s = float(g_out.item()) if g_out.ndim == 0 else float(g_out)
        return s * gamma, s * xi_sum, s * grad_pi0

    return hmm_grad_impl

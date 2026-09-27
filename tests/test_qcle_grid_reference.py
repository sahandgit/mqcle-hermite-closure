"""Independent real-component QCLE grid solver and cross-code agreement."""
from __future__ import annotations

import numpy as np
import scipy.linalg

from mqcle.coefficient_operator import Grid, coefficient_rhs, physical_matrices
from mqcle.errors import qcle_grid_error_metrics
from mqcle.models import ConstantCouplingModel, TullyDualAvoidedCrossing
from mqcle.propagation import rk4_step
from mqcle.qcle_grid_reference import (IndependentQCLEGridSolver,
                                       QCLEGridParams, QCLEGridState)
from mqcle.types import TraceConvention


def _solver(model=None, n_R=32, n_P=24):
    model = model or TullyDualAvoidedCrossing()
    params = QCLEGridParams(-8.0, 8.0, n_R, -4.0, 8.0, n_P,
                            mass=model.mass, fft_workers=1)
    return IndependentQCLEGridSolver(model, params)


def _coefficient_grid(solver):
    p = solver.params
    return Grid(solver.R, solver.P, solver.dR, solver.dP,
                solver.kR, solver.kP,
                p.R_max - p.R_min, p.P_max - p.P_min)


def _smooth_state(solver):
    RR, PP = solver.mesh()
    x = 2.0 * np.pi * (RR - solver.params.R_min) \
        / (solver.params.R_max - solver.params.R_min)
    y = 2.0 * np.pi * (PP - solver.params.P_min) \
        / (solver.params.P_max - solver.params.P_min)
    return QCLEGridState(
        1.0 + 0.10 * np.cos(x) + 0.08 * np.sin(y),
        0.7 + 0.05 * np.sin(2.0 * x) * np.cos(y),
        0.03 * np.cos(x) * np.cos(2.0 * y),
        0.02 * np.sin(x) * np.sin(y),
    )


def test_real_fft_derivatives_are_exact_for_resolved_harmonics():
    solver = _solver()
    RR, PP = solver.mesh()
    LR = solver.params.R_max - solver.params.R_min
    LP = solver.params.P_max - solver.params.P_min
    f = np.sin(2.0 * np.pi * 3.0 * (RR - solver.params.R_min) / LR) \
        * np.cos(2.0 * np.pi * 2.0 * (PP - solver.params.P_min) / LP)
    dR = (2.0 * np.pi * 3.0 / LR) \
        * np.cos(2.0 * np.pi * 3.0 * (RR - solver.params.R_min) / LR) \
        * np.cos(2.0 * np.pi * 2.0 * (PP - solver.params.P_min) / LP)
    dP = -(2.0 * np.pi * 2.0 / LP) \
        * np.sin(2.0 * np.pi * 3.0 * (RR - solver.params.R_min) / LR) \
        * np.sin(2.0 * np.pi * 2.0 * (PP - solver.params.P_min) / LP)
    assert np.max(np.abs(solver._dR(f) - dR)) < 2e-14
    assert np.max(np.abs(solver._dP(f) - dP)) < 2e-14


def test_independent_real_rhs_matches_closed_coefficient_rhs():
    model = TullyDualAvoidedCrossing()
    solver = _solver(model)
    state = _smooth_state(solver)
    reference = coefficient_rhs(state.as_matrix(), _coefficient_grid(solver), model)
    independent = solver.rhs_state(state).as_matrix()
    assert np.max(np.abs(independent - reference)) < 5e-15


def test_independent_rk4_step_matches_complex_coefficient_rk4():
    model = TullyDualAvoidedCrossing()
    solver = _solver(model)
    grid = _coefficient_grid(solver)
    state = _smooth_state(solver)
    cache = physical_matrices(model, grid, TraceConvention.TRACELESS)
    ref = rk4_step(state.as_matrix(), 0.1, grid, model, cache=cache)
    got = solver.step(state, 0.1).as_matrix()
    assert np.max(np.abs(got - ref)) < 5e-15


def test_independent_rk4_has_fourth_order_temporal_convergence():
    model = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005,
                                  k=0.0, mass=2000.0)
    solver = _solver(model, 16, 16)
    area = solver.cell_area
    matrix0 = np.array([[0.6, 0.25 + 0.15j],
                        [0.25 - 0.15j, 0.4]], dtype=complex)
    rho0 = np.broadcast_to(matrix0, solver.shape + (2, 2)).copy()
    state0 = QCLEGridState.from_matrix(rho0)
    h = model.h(np.zeros((1, 1)))[0]
    total_time = 80.0
    U = scipy.linalg.expm(-1j * h * total_time)
    exact = U @ matrix0 @ U.conj().T

    errors = []
    for dt in (4.0, 2.0, 1.0):
        state = state0.copy()
        for _ in range(int(round(total_time / dt))):
            state = solver.step(state, dt)
        integrated = np.sum(state.as_matrix(), axis=(0, 1)) * area
        target = exact * solver.shape[0] * solver.shape[1] * area
        errors.append(np.max(np.abs(integrated - target)))
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all((orders > 3.8) & (orders < 4.2)), orders


def test_qcle_grid_error_metrics_resolve_field_and_observable_errors():
    solver = _solver()
    ref = _smooth_state(solver).as_matrix()
    zero = qcle_grid_error_metrics(ref, ref, solver.cell_area)
    assert all(np.allclose(value, 0.0) for value in zero.values())

    got = ref.copy()
    got[..., 0, 0] += 1e-3
    measured = qcle_grid_error_metrics(ref, got, solver.cell_area)
    expected_population = 1e-3 * np.prod(solver.shape) * solver.cell_area
    assert np.isclose(measured["population_linf"], expected_population)
    assert measured["field_relative_l2"] > 0.0
    assert measured["trace_density_l1"] > 0.0

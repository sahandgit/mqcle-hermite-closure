r"""Module 7 -- original-space excess Liouvillian.

Density-side operator
---------------------
.. math::
    \mathcal A'\rho=-\frac{\hbar}{8}\sum_{\lambda\lambda'}
    \nabla_R\mathsf h^{\lambda\lambda'}(R)\cdot D_{\lambda\lambda'}\nabla_P\rho,
    \qquad
    D_{\lambda\lambda'}=\partial_{r_{\lambda'}}\partial_{r_\lambda}
      +\partial_{p_{\lambda'}}\partial_{p_\lambda}.

Sign and prefactor
------------------
Kim, Nassimi & Kapral, J. Chem. Phys. 129, 084102 (2008), Eq. (30), give the
operator for *dynamical variables* with the prefactor :math:`+\hbar/8`.  The
density obeys the formal adjoint with respect to :math:`dx\,dX`; the operator
carries three derivatives whose coefficient depends on :math:`R` alone, so
integrating by parts three times contributes :math:`(-1)^3` and the
density-side prefactor is :math:`-\hbar/8`.  That is the convention of Nassimi,
Bonella & Kapral, J. Chem. Phys. 133, 134115 (2010), Eq. (10), and of this
project's existing operator implementation.

Which matrix appears
--------------------
:math:`\mathsf h` is whatever :func:`mqcle.mapping.subsystem_split` returned for
the active :class:`~mqcle.types.TraceConvention` -- the *same* matrix that
enters :math:`H_m`.  The API makes it impossible to use one matrix here and
another in the Hamiltonian; see ``CORRECTIONS.md`` entry C2 for why that
matters.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .constants import HBAR
from .fields import FiniteDifferenceField, ScalarField
from .mapping import subsystem_split
from .models import TwoStateModel
from .types import TraceConvention

__all__ = ["excess_operator_on_field", "excess_operator_on_function",
           "excess_coefficients"]


def excess_coefficients(model: TwoStateModel, R, hbar: float = HBAR,
                        convention: TraceConvention = TraceConvention.TRACELESS
                        ) -> np.ndarray:
    r"""The tensor :math:`-\frac{\hbar}{8}\nabla_R\mathsf h^{\lambda\lambda'}`,
    shape ``(..., Nb, N, N)``.

    Exposed separately because it is the entire model dependence of the excess
    term: it vanishes identically for an R-independent subsystem Hamiltonian,
    and its magnitude sets the rate at which the Poisson-bracket approximation
    leaves the physical subspace.
    """
    sp_ = subsystem_split(model, R, convention)
    return -(hbar / 8.0) * sp_.dh_map_dR


def excess_operator_on_field(field: ScalarField, x, R, P, model: TwoStateModel,
                             hbar: float = HBAR,
                             convention: TraceConvention = TraceConvention.TRACELESS
                             ) -> np.ndarray:
    r"""Apply :math:`\mathcal A'` to a :class:`~mqcle.fields.ScalarField`."""
    coeff = excess_coefficients(model, R, hbar, convention)   # (..., Nb, N, N)
    tensor = field.excess_tensor(x, R, P)                     # (..., Nb, N, N)
    return np.einsum("...kij,...kij->...", coeff, tensor)


def excess_operator_on_function(fn: Callable, x, R, P, model: TwoStateModel,
                                hbar: float = HBAR,
                                convention: TraceConvention = TraceConvention.TRACELESS,
                                **fd_kwargs) -> np.ndarray:
    """Same, for a plain callable ``fn(x, R, P)``, differentiated numerically."""
    field = FiniteDifferenceField(fn, model.nstates, model.nbath, **fd_kwargs)
    return excess_operator_on_field(field, x, R, P, model, hbar, convention)

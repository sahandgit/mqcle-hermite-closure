r"""Module 8 -- the full original-space MQCLE generator.

.. math:: \mathcal A_m=\mathcal A_{\rm PB}+\mathcal A' .

The two contributions are computed by :mod:`mqcle.pbme` and
:mod:`mqcle.excess` independently and are *never* merged algebraically, so the
code can always report them separately (PART XII).  Retaining only
:math:`\mathcal A_{\rm PB}` is the Poisson-bracket mapping equation; that
object is called PBME everywhere in this codebase and never "the MQCLE"
(PART II rule 5).
"""
from __future__ import annotations

import numpy as np

from .constants import HBAR
from .excess import excess_operator_on_field
from .fields import FiniteDifferenceField, ScalarField
from .models import TwoStateModel
from .pbme import pbme_generator_on_field
from .types import OperatorParts, TraceConvention

__all__ = ["full_generator_on_field", "full_generator_on_function"]


def full_generator_on_field(field: ScalarField, x, R, P, model: TwoStateModel,
                            hbar: float = HBAR, mass: float | None = None,
                            convention: TraceConvention = TraceConvention.TRACELESS,
                            form: str = "poisson") -> OperatorParts:
    """Return the PBME and excess contributions separately.

    ``OperatorParts.total`` is their sum; nothing else in the codebase adds
    them, so a caller can always attribute a result to one term or the other.
    """
    pb = pbme_generator_on_field(field, x, R, P, model, hbar, mass,
                                 convention, form=form)
    ex = excess_operator_on_field(field, x, R, P, model, hbar, convention)
    return OperatorParts(pbme=np.asarray(pb), excess=np.asarray(ex))


def full_generator_on_function(fn, x, R, P, model: TwoStateModel,
                               hbar: float = HBAR, mass: float | None = None,
                               convention: TraceConvention = TraceConvention.TRACELESS,
                               **fd_kwargs) -> OperatorParts:
    """Same, for a plain callable ``fn(x, R, P)``, differentiated numerically."""
    field = FiniteDifferenceField(fn, model.nstates, model.nbath, **fd_kwargs)
    return full_generator_on_field(field, x, R, P, model, hbar, mass, convention)

"""Authoritative production settings for the Tully-II momentum family.

The six momenta are used by every comparison consumer.  Keeping their grids,
boxes, final times, and manuscript tags here prevents a new regime from being
added to one branch while silently remaining absent from another.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MomentumRegime", "MOMENTA", "REGIMES", "TAGS", "regime_for"]


@dataclass(frozen=True)
class MomentumRegime:
    P0: int
    R_box: tuple[float, float]
    n_R: int
    P_center: float
    P_half: float
    n_P: int
    dt: float
    T: float
    se_n_R: int
    se_dt: float = 0.125

    @property
    def P_box(self) -> tuple[float, float]:
        return self.P_center - self.P_half, self.P_center + self.P_half

    @property
    def sigma_R(self) -> float:
        return 10.0 / self.P0

    @property
    def shear_P_scale(self) -> float:
        return 2000.0 * self.sigma_R / self.T

    def coefficient_defaults(self) -> dict:
        return {
            "R_box": self.R_box,
            "n_R": self.n_R,
            "P_center": self.P_center,
            "P_half": self.P_half,
            "n_P": self.n_P,
            "dt": self.dt,
            "T": self.T,
            "se_n_R": self.se_n_R,
            "se_dt": self.se_dt,
        }


# dP <= 0.1 in every case, i.e. <= 0.3 of the phase-space shear scale 1/3.
# The R meshes resolve the narrowest initial packet at dR/sigma_R <= 0.36.
# P0=10 and 20 use wider momentum boxes because low-energy reflection creates
# a negative-P branch.  P0=10 additionally needs 4096x2048: the 2048x1024
# attempt developed equal, cell-alternating contamination at both R edges and
# later at both P edges.  P0=70 uses T=857.5 so T/dt is an integer (1715).
REGIMES: dict[int, MomentumRegime] = {
    10: MomentumRegime(10, (-45.0, 45.0), 4096, 10.0, 40.0, 2048,
                       0.5, 6000.0, 8192),
    20: MomentumRegime(20, (-45.0, 45.0), 4096, 20.0, 40.0, 1024,
                       0.5, 3000.0, 8192),
    30: MomentumRegime(30, (-45.0, 45.0), 1024, 30.0, 22.0, 512,
                       0.5, 2000.0, 8192),
    40: MomentumRegime(40, (-45.0, 45.0), 1024, 40.0, 22.0, 512,
                       0.5, 1500.0, 8192),
    70: MomentumRegime(70, (-35.0, 35.0), 2048, 70.0, 32.0, 1024,
                       0.5, 857.5, 16384),
    100: MomentumRegime(100, (-35.0, 35.0), 2048, 100.0, 40.0, 1024,
                        0.5, 600.0, 16384),
}

MOMENTA = tuple(REGIMES)
TAGS = {
    10: "PTen",
    20: "PTwenty",
    30: "PThirty",
    40: "PForty",
    70: "PSeventy",
    100: "POneHundred",
}


def regime_for(P0: float) -> MomentumRegime | None:
    """Return the registered regime when ``P0`` is an exact listed momentum."""
    rounded = int(round(float(P0)))
    if abs(float(P0) - rounded) > 1e-12:
        return None
    return REGIMES.get(rounded)

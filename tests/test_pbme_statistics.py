"""Independent-seed, block-SEM, and confidence-interval checks."""
from __future__ import annotations

import numpy as np

from mqcle.propagation import block_statistics, sample_projected_ensemble


def test_projected_sampling_seed_is_reproducible():
    kw = dict(nstates=2, init_state=0, R0=-15.0, P0=40.0,
              sigma_R=0.25, hbar=1.0)
    a = sample_projected_ensemble(np.random.default_rng(101), 1000, **kw)
    b = sample_projected_ensemble(np.random.default_rng(101), 1000, **kw)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.R, b.R)
    assert np.array_equal(a.P, b.P)
    assert np.array_equal(a.weights, b.weights)


def test_sem_and_ci_use_actual_block_variance():
    B = 64
    per_block = np.tile(np.array([-1.0, 1.0]), B // 2)
    blocks = (per_block / B)[None, :, None]
    st = block_statistics(blocks)
    expected_sd = per_block.std(ddof=1)
    assert np.isclose(st["sd"][0, 0], expected_sd)
    assert np.isclose(st["sem"][0, 0], expected_sd / np.sqrt(B))
    assert np.isclose(st["ci_hi"][0, 0] - st["mean"][0, 0],
                      1.96 * st["sem"][0, 0])


def test_sem_decreases_with_nested_sample_size():
    B = 64
    per_block = np.tile(np.array([-1.0, 1.0]), B // 2)
    blocks = (per_block / B)[None, :, None]
    sem8 = block_statistics(blocks, 8)["sem"][0, 0]
    sem64 = block_statistics(blocks, 64)["sem"][0, 0]
    assert sem64 < sem8

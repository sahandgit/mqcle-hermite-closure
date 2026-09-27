r"""Optimised evaluation of the *same* finite Hermite generator.

Nothing in this module changes the mathematics.  The operator applied is
:math:`\mathbb L_H` of Eq. (L-H) exactly as written in
:func:`mqcle.hermite.operator.hermite_rhs`,

.. math::

    \mathbb L_H[C]=-\frac{P}{M}D_RC+\mathsf R(R)C+\mathsf G(R)D_PC ,

on the same grid, with the same Fourier spectral derivatives (same Nyquist
convention), the same matrices :math:`\mathsf R,\mathsf G`, and the same
classical Runge--Kutta tableau.  Only the *arithmetic realisation* differs.
Four changes, each of which is an identity rather than an approximation:

1. **Real arithmetic.**  :math:`\mathsf R` and :math:`\mathsf G` are exactly
   real: :math:`\mathbb H` is the real form of the subsystem Hamiltonian, so
   :math:`K=\hbar^{-1}\mathbb J\mathbb H` is real, the :math:`\mathcal T_{ij}`
   matrices are real, and :math:`\partial_RV_b` is real.  The advection
   coefficient :math:`-P/M` is real and the Fourier derivative of a real
   periodic field is real once the Nyquist mode is dropped -- which the
   validated routine already does, and for exactly this reason.  Hence
   :math:`\mathbb L_H` maps real fields to real fields.  The physical-frame
   initial datum :math:`C=B\,\mathrm{vec}(\rho_W)` is real for Hermitian
   :math:`\rho_W`, so the entire trajectory is real and the ``complex128``
   path is carrying an identically zero imaginary part through every
   transform and every byte of memory.  Storing the state as ``float64``
   halves the memory and the bandwidth, and lets the transforms be
   real-to-complex.  :func:`generator_is_real` and :func:`state_is_real`
   check the hypothesis rather than assuming it, and the caller must fall
   back to the reference path when either fails.

2. **Cache-blocked spectral derivatives.**  The :math:`R` transform runs on
   the strided axis of the mode-major layout.  Transforming
   ``p_slab``-wide column slabs through a contiguous buffer is
   bit-for-bit identical and measurably faster, because one length-``n_R``
   transform then touches a working set that fits in cache.

3. **Dense batched GEMM for the mode contraction.**  The reference walks 51
   nonzero blocks, each a full-plane AXPY through a temporary.  Contracting
   the dense ``(M, M)`` matrix at each ``R`` instead reads the field once and
   writes it once; the extra arithmetic is free on a bandwidth-bound kernel.
   This is the one step that is not bit-identical: it reorders a sum of
   floating-point numbers, so it agrees to round-off, not to the last bit.

4. **Fused stage updates and preallocated buffers.**  The Runge--Kutta
   combinations are applied to each :math:`R` chunk while that chunk is still
   in cache, so no stage derivative is ever materialised as a full array and
   nothing is allocated inside the time loop.

The reference implementation in :mod:`mqcle.hermite.operator` and
:mod:`mqcle.hermite.propagator` is untouched and remains the definition of
the method; :mod:`tests.test_gaussian_hermite` gate GH-12 holds this module
to it.
"""
from __future__ import annotations

import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft
from scipy.sparse.linalg import LinearOperator, expm_multiply

__all__ = ["RealHermiteKernel", "generator_is_real", "state_is_real",
           "rk4_real", "krylov_real", "single_threaded_blas",
           "limit_blas_threads_via_env"]


# --------------------------------------------------------------------------
# thread bookkeeping
# --------------------------------------------------------------------------
# The kernel parallelises over modes and over R chunks itself.  If BLAS also
# threads inside each of those workers the machine is oversubscribed, and the
# batched GEMM here is small enough that BLAS threading loses even without
# nesting: measured on two cores, 0.045 s with one BLAS thread against 0.057 s
# with two, for the same chunked contraction.  So BLAS is pinned to one
# thread.  Environment variables are the portable mechanism and must be set
# before numpy is imported, which is why a *script* calls
# ``limit_blas_threads_via_env`` at its top; ``single_threaded_blas`` is the
# fallback for a process where numpy is already loaded.
#: DGKS threshold.  A first projection that leaves less than this fraction of
#: the incoming norm has suffered cancellation and is repeated.  1/sqrt(2) is
#: the classical choice and is what the stability analysis assumes.
_DGKS_ETA = 0.7071067811865476

_BLAS_ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS")


def limit_blas_threads_via_env(n: int = 1) -> None:
    """Pin BLAS to ``n`` threads.  Call before ``import numpy``."""
    for var in _BLAS_ENV_VARS:
        os.environ.setdefault(var, str(int(n)))


@contextlib.contextmanager
def single_threaded_blas():
    """Pin BLAS to one thread for the duration, if ``threadpoolctl`` is here."""
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        yield None
        return
    with threadpool_limits(limits=1, user_api="blas") as ctl:
        yield ctl


# --------------------------------------------------------------------------
# validity of the real realisation
# --------------------------------------------------------------------------
def generator_is_real(gen) -> bool:
    r"""``True`` when :math:`\mathsf R` and :math:`\mathsf G` are exactly real.

    Exactly, not approximately: a nonzero imaginary part of any size means the
    hypothesis behind the real path does not hold and the reference complex
    path must be used.
    """
    return (np.max(np.abs(np.imag(gen.Rmat))) == 0.0
            and np.max(np.abs(np.imag(gen.Gmat))) == 0.0)


def state_is_real(C, tol: float = 0.0) -> bool:
    r"""``True`` when the coefficient field carries no imaginary part.

    ``tol`` is absolute.  The default is exact: a physical-frame state built
    as :math:`C=B\,\mathrm{vec}(\rho_W)` from a Hermitian :math:`\rho_W` has
    an imaginary part that is identically zero, not merely small.
    """
    C = np.asarray(C)
    if not np.iscomplexobj(C):
        return True
    return float(np.max(np.abs(C.imag))) <= tol


# --------------------------------------------------------------------------
class RealHermiteKernel:
    r"""Real, blocked, threaded application of :math:`\mathbb L_H`.

    Parameters
    ----------
    gen
        A built :class:`mqcle.hermite.propagator.HermiteGenerator`.  Its
        ``Rmat``/``Gmat`` must be exactly real; see :func:`generator_is_real`.
    nthread
        Worker threads.  ``None`` takes ``os.cpu_count()``.  The transforms
        and BLAS calls release the GIL, so these are real threads.
    r_slab
        Column width of the :math:`R`-transform blocking.  Chosen so that
        ``n_R * r_slab * 8`` bytes sit in L2.
    p_slab
        Row height of the :math:`P`-transform blocking.
    chunk
        Number of :math:`R` points per GEMM batch.

    Notes
    -----
    The kernel owns its scratch arrays, so an instance is **not** safe to use
    from two threads at once.  ``apply`` is internally threaded instead.
    """

    def __init__(self, gen, nthread: int | None = None, r_slab: int = 64,
                 p_slab: int = 256, chunk: int = 8):
        if not generator_is_real(gen):
            raise ValueError(
                "RealHermiteKernel: the generator is not exactly real; "
                "use the reference complex path in mqcle.hermite.propagator")
        g = gen.grid
        self.gen, self.grid = gen, g
        self.n_R, self.n_P = g.shape
        self.M = len(gen.modes)

        self.Rm = np.ascontiguousarray(np.real(gen.Rmat))
        self.Gm = np.ascontiguousarray(np.real(gen.Gmat))
        self.RmT = np.ascontiguousarray(self.Rm.transpose(0, 2, 1))
        self.GmT_neg = np.ascontiguousarray(-self.Gm.transpose(0, 2, 1))
        self.adv = np.ascontiguousarray(-(np.asarray(g.P, float)
                                          / float(gen.mass)))[None, None, :]

        self.facR = self._half_wavenumbers(g.kR, self.n_R)[:, None]
        self.facP = self._half_wavenumbers(g.kP, self.n_P)[None, :]

        # modes that no G block reads: their D_P is never used, so it is not
        # computed.  Assert rather than trust: the corresponding columns of G
        # must be identically zero for the dense contraction to agree.
        self.need_dP = sorted({b[1] for b in gen._G_blocks})
        dead = [a for a in range(self.M) if a not in self.need_dP]
        if dead and np.max(np.abs(self.Gm[:, :, dead])) != 0.0:
            raise ValueError("RealHermiteKernel: G has a nonzero column for a "
                             "mode absent from its sparse block list")

        self.r_slab = int(r_slab)
        self.p_slab = int(p_slab)
        self.chunk = int(chunk)
        self.nthread = int(nthread or os.cpu_count() or 1)
        self._pool = ThreadPoolExecutor(self.nthread)
        self._blas_env = {v: os.environ.get(v) for v in _BLAS_ENV_VARS}
        self._blas_pinned = self._blas_env.get("OPENBLAS_NUM_THREADS") == "1"
        # Entered once and held, not per application: re-entering it four
        # times a step would be measurable overhead for no benefit.
        self._blas_ctx = None
        if not self._blas_pinned:
            self._blas_ctx = single_threaded_blas()
            self._blas_ctx.__enter__()

        shape = (self.M, self.n_R, self.n_P)
        self.dR = np.empty(shape, dtype=np.float64)
        self.dP = np.zeros(shape, dtype=np.float64)          # dead rows stay 0
        self._local = threading.local()

        # task lists are fixed for the life of the kernel
        self._dtasks = ([("R", a, p) for a in range(self.M)
                         for p in range(0, self.n_P, self.r_slab)]
                        + [("P", a, r) for a in self.need_dP
                           for r in range(0, self.n_R, self.p_slab)])
        # the adjoint needs D_P for every mode; see apply(adjoint=True)
        self._dtasks_all = ([("R", a, p) for a in range(self.M)
                             for p in range(0, self.n_P, self.r_slab)]
                            + [("P", a, r) for a in range(self.M)
                               for r in range(0, self.n_R, self.p_slab)])
        # Two independent block sizes, which an earlier version conflated:
        # ``chunk`` is the cache tile of the contraction and is a property of
        # the machine's L2, while ``blk`` is the unit of work handed to a
        # thread and is a property of the thread count.  Tying the tile to the
        # thread count silently disabled the tiling on one thread.  Several
        # blocks per thread so that a straggler cannot stall the phase.
        per_thread = max(1, self.n_R // (4 * self.nthread))
        blk = max(self.chunk,
                  (per_thread // self.chunk) * self.chunk)
        self._ctasks = [(s, min(s + blk, self.n_R))
                        for s in range(0, self.n_R, blk)]

    # ----------------------------------------------------------------------
    @staticmethod
    def _half_wavenumbers(k, n: int) -> np.ndarray:
        r"""``i k`` on the ``rfft`` half spectrum, Nyquist dropped.

        Identical to the convention of
        :func:`mqcle.coefficient_operator.spectral_derivative`, which zeroes
        ``k[n//2]`` for even ``n`` because that mode has no well-defined
        derivative.
        """
        kk = np.asarray(k, dtype=float).copy()
        if n % 2 == 0:
            kk[n // 2] = 0.0
        return 1j * kk[:n // 2 + 1]

    def _buf(self) -> np.ndarray:
        """Per-thread contiguous staging buffer for one ``R`` column slab."""
        buf = getattr(self._local, "rbuf", None)
        if buf is None:
            buf = np.empty((self.n_R, self.r_slab))
            self._local.rbuf = buf
        return buf

    @property
    def state_bytes(self) -> int:
        return self.M * self.n_R * self.n_P * 8

    # ----------------------------------------------------------------------
    def apply(self, C: np.ndarray, out: np.ndarray | None = None,
              update=None, adjoint: bool = False) -> np.ndarray | None:
        r"""Apply :math:`\mathbb L_H` to the real field ``C``.

        With ``out`` the result is written there.  With ``update`` the result
        is never materialised: ``update(slice_, acc)`` is called for each
        :math:`R` chunk with ``acc`` of shape ``(n_chunk, M, n_P)``, while the
        chunk is still in cache.  Exactly one of the two must be given.

        ``adjoint`` applies :math:`\mathbb L_H^{\mathsf T}` instead.  The
        Fourier spectral derivatives are skew on a periodic grid once the
        Nyquist mode is dropped, :math:`D^{\mathsf T}=-D`, and both
        :math:`\mathsf R` and :math:`\mathsf G` are :math:`R`-local, so

        .. math::

            \mathbb L_H^{\mathsf T}C
            =+\frac{P}{M}D_RC+\mathsf R^{\mathsf T}C
             -\mathsf G^{\mathsf T}D_PC .

        It is needed by the one-norm estimator inside the Krylov exponential,
        not by the time stepping, so it is written for correctness rather than
        for speed: every mode's :math:`D_P` is computed, because the zero
        columns of :math:`\mathsf G` are not the zero columns of
        :math:`\mathsf G^{\mathsf T}`.
        """
        if (out is None) == (update is None):
            raise ValueError("RealHermiteKernel.apply: give out or update")
        n_R, n_P = self.n_R, self.n_P
        dR, dP = self.dR, self.dP
        facR, facP, rs, ps = self.facR, self.facP, self.r_slab, self.p_slab
        dtasks = self._dtasks_all if adjoint else self._dtasks

        def derivative(task):
            axis, a, off = task
            if axis == "R":
                w = min(rs, n_P - off)
                buf = self._buf()
                blk = buf if w == rs else np.empty((n_R, w))
                blk[:, :w] = C[a, :, off:off + w]
                S = sfft.rfft(blk[:, :w], axis=0)
                S *= facR
                dR[a, :, off:off + w] = sfft.irfft(S, n=n_R, axis=0,
                                                   overwrite_x=True)
            else:
                e = min(off + ps, n_R)
                S = sfft.rfft(C[a, off:e], axis=1)
                S *= facP
                dP[a, off:e] = sfft.irfft(S, n=n_P, axis=1, overwrite_x=True)

        list(self._pool.map(derivative, dtasks))

        Cv = np.moveaxis(C, 0, 1)
        dPv = np.moveaxis(dP, 0, 1)
        dRv = np.moveaxis(dR, 0, 1)
        ov = None if out is None else np.moveaxis(out, 0, 1)
        if adjoint:
            Rm, Gm, adv = self.RmT, self.GmT_neg, -self.adv
        else:
            Rm, Gm, adv = self.Rm, self.Gm, self.adv

        nc = self.chunk

        def contract(se):
            lo, hi = se
            for s in range(lo, hi, nc):
                e = min(s + nc, hi)
                acc = np.matmul(Rm[s:e], Cv[s:e])
                acc += np.matmul(Gm[s:e], dPv[s:e])
                acc += dRv[s:e] * adv
                if ov is not None:
                    ov[s:e] = acc
                else:
                    update(slice(s, e), acc)

        list(self._pool.map(contract, self._ctasks))
        return out

    # convenience: the same signature as HermiteGenerator.rhs
    def rhs(self, C: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty_like(C)
        return self.apply(C, out=out)

    def rhs_adjoint(self, C: np.ndarray,
                    out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty_like(C)
        return self.apply(C, out=out, adjoint=True)

    def as_linear_operator(self) -> LinearOperator:
        r"""Matrix-free real ``LinearOperator``; the global matrix never forms.

        This is the object Eq. (Krylov-step) acts with.  ``rmatvec`` is the
        transpose of :meth:`apply`, which the one-norm estimator inside
        :func:`scipy.sparse.linalg.expm_multiply` requires.
        """
        shape = (self.M, self.n_R, self.n_P)
        n = self.M * self.n_R * self.n_P
        self.matvec_count = 0
        self.rmatvec_count = 0

        def matvec(v):
            self.matvec_count += 1
            out = np.empty(shape)
            self.apply(np.ascontiguousarray(v).reshape(shape), out=out)
            return out.reshape(n)

        def rmatvec(v):
            self.rmatvec_count += 1
            out = np.empty(shape)
            self.apply(np.ascontiguousarray(v).reshape(shape), out=out,
                       adjoint=True)
            return out.reshape(n)

        return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec,
                              dtype=np.float64)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if self._blas_ctx is not None:
            self._blas_ctx.__exit__(None, None, None)
            self._blas_ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------
def rk4_real(kern: RealHermiteKernel, C: np.ndarray, dt: float, n_steps: int,
             save_every: int = 1, callback=None) -> np.ndarray:
    r"""Classical RK4 on ``kern``, three live full-size registers.

    The tableau is the classical one used by
    :func:`mqcle.hermite.propagator.propagate_rk4` and by
    ``scripts/07g_gh_coefficient_propagation.py``:

    .. math::

        C^{n+1}=C^n+\tfrac{\Delta t}{6}(k_1+2k_2+2k_3+k_4),

    with the stage arguments :math:`C^n`, :math:`C^n+\tfrac{\Delta t}2k_1`,
    :math:`C^n+\tfrac{\Delta t}2k_2`, :math:`C^n+\Delta t\,k_3`.  The stage
    derivatives are consumed chunk by chunk instead of being stored, so the
    working set is ``C``, the accumulator and the stage argument, plus the two
    derivative scratch arrays the kernel owns.

    ``callback(step, t, C)`` runs after each saved step.  ``C`` is modified
    in place and also returned.
    """
    C = np.ascontiguousarray(C, dtype=np.float64)
    acc = np.empty_like(C)          # running sum  k1 + 2k2 + 2k3 + k4
    stage = np.empty_like(C)        # argument of the next kernel call

    if callback is not None:
        callback(0, 0.0, C)

    for step in range(1, n_steps + 1):
        Cv = np.moveaxis(C, 0, 1)
        av = np.moveaxis(acc, 0, 1)
        sv = np.moveaxis(stage, 0, 1)

        # The stage combination is accumulated in the reference's own order,
        # (((k1 + 2 k2) + 2 k3) + k4) scaled by dt/6 and added to C last, so
        # that only the mode contraction differs in its floating-point
        # rounding from mqcle.hermite.propagator.propagate_rk4.
        def u1(sl, k, Cv=Cv, av=av, sv=sv):
            av[sl] = k
            sv[sl] = Cv[sl] + 0.5 * dt * k

        def u2(sl, k, Cv=Cv, av=av, sv=sv):
            av[sl] += 2.0 * k
            sv[sl] = Cv[sl] + 0.5 * dt * k

        def u3(sl, k, Cv=Cv, av=av, sv=sv):
            av[sl] += 2.0 * k
            sv[sl] = Cv[sl] + dt * k

        def u4(sl, k, Cv=Cv, av=av):
            av[sl] = Cv[sl] + (dt / 6.0) * (av[sl] + k)

        kern.apply(C, update=u1)
        kern.apply(stage, update=u2)
        kern.apply(stage, update=u3)
        kern.apply(stage, update=u4)
        C, acc = acc, C             # swap: no copy

        if callback is not None and (step % save_every == 0 or step == n_steps):
            callback(step, step * dt, C)
    return C


# --------------------------------------------------------------------------
def krylov_real(kern: RealHermiteKernel, C: np.ndarray, times,
                callback=None) -> np.ndarray:
    r"""Matrix-free exponential action, Eq. (Krylov-step), on the real kernel.

    Advances :math:`C(t+\Delta t)=\exp(\Delta t\,\mathbb L_H)C(t)` one saved
    interval at a time through :func:`scipy.sparse.linalg.expm_multiply` on the
    matrix-free operator of
    :meth:`RealHermiteKernel.as_linear_operator`.  The global matrix is never
    assembled; the only thing the exponential ever sees is the action
    :math:`Y=\mathbb L_HX`.

    The generator is exactly traceless.  The two bath-derivative channels have
    zero diagonal on a symmetric spectral grid, and
    :math:`\mathrm{tr}\,\mathsf R=-\sum_iK_{ii}n_i=0` because :math:`K` is
    antisymmetric, so ``traceA=0`` is supplied exactly rather than estimated.

    ``callback(index, t, C)`` runs at each saved time, including ``t=0``.
    Returns the terminal state; intermediate states are not retained, because
    at production sizes the trajectory does not fit in memory.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("krylov_real: times must be a 1-D array")
    if np.any(np.diff(times) < 0):
        raise ValueError("krylov_real: times must be nondecreasing")

    A = kern.as_linear_operator()
    shape = (kern.M, kern.n_R, kern.n_P)
    v = np.ascontiguousarray(C, dtype=np.float64).ravel().copy()

    if callback is not None:
        callback(0, 0.0, v.reshape(shape))

    t_prev = 0.0
    for i, t in enumerate(times):
        dt = float(t) - t_prev
        if dt > 0:
            v = expm_multiply(A, v, start=0.0, stop=dt, num=2,
                              endpoint=True, traceA=0.0)[-1]
        t_prev = float(t)
        if callback is not None:
            callback(i + 1, float(t), v.reshape(shape))
    return v.reshape(shape)


# --------------------------------------------------------------------------
# matrix-free Arnoldi exponential action
# --------------------------------------------------------------------------
def arnoldi_expmv(apply, v: np.ndarray, t: float, m: int = 8,
                  tol: float = 1e-10, work=None, stats: dict | None = None
                  ) -> np.ndarray:
    r"""One matrix-free Arnoldi exponential action, Eq. (Krylov-step).

    Computes :math:`v\mapsto e^{t\mathbb L_H}v` by building a Krylov subspace

    .. math::

        \mathcal K_m(\mathbb L_H,v)
        =\mathrm{span}\{v,\mathbb L_Hv,\dots,\mathbb L_H^{m-1}v\},

    reducing :math:`\mathbb L_H` to the :math:`m\times m` upper Hessenberg
    :math:`H_m=V_m^{\mathsf T}\mathbb L_HV_m`, and taking
    :math:`\beta V_m e^{\tau H_m}e_1`.  The global matrix is never formed; the
    only thing this routine ever does with the operator is ``apply(x, out)``.

    Substepping is adaptive.  With
    :math:`h_{m+1,m}` the last Arnoldi coefficient, Saad's a posteriori
    estimate of the local error is

    .. math::

        \mathrm{err}\simeq\beta\,h_{m+1,m}\,
        \bigl|\bigl(e^{\tau H_m}\bigr)_{m,1}\bigr| ,

    which is evaluated *before* the step is accepted, so a rejected
    :math:`\tau` costs the small dense exponential and not another Arnoldi
    cycle.  The basis is rebuilt at each accepted substep, which is what keeps
    the storage at :math:`m+1` state vectors rather than growing with time.

    Parameters
    ----------
    apply
        ``apply(x, out)`` writing :math:`\mathbb L_Hx` into ``out``.
    m
        Krylov dimension.  Cost per substep is ``m`` operator applications and
        storage is ``m + 1`` state vectors, so at production sizes ``m`` is
        bounded by memory, not by accuracy: see the module notes.
    tol
        Requested local error per substep, relative to :math:`\|v\|`.
    work
        Optional preallocated ``(m + 1,) + v.shape`` basis, reused across
        calls so that a time loop allocates nothing.
    stats
        Optional dict; receives ``applications``, ``substeps``, ``rejected``
        and ``breakdowns``.
    """
    from scipy.linalg import expm as _dense_expm

    v = np.ascontiguousarray(v, dtype=np.float64)
    shape = v.shape
    if work is None:
        work = np.empty((m + 1,) + shape)
    V = work
    H = np.zeros((m + 1, m))
    napp = nsub = nrej = nbrk = nreortho = 0

    t_now = 0.0
    tau = float(t)
    v = v.copy()
    while t_now < t - 1e-14 * max(1.0, abs(t)):
        beta = float(np.sqrt(np.sum(v * v)))
        if beta == 0.0:
            break
        np.multiply(v, 1.0 / beta, out=V[0])
        H[:] = 0.0
        k_eff = m
        Vflat = V.reshape(m + 1, -1)
        for j in range(m):
            apply(V[j], out=V[j + 1])
            napp += 1
            w = Vflat[j + 1]
            # Classical Gram-Schmidt with DGKS selective re-orthogonalisation.
            # Two BLAS-2 products per pass rather than the O(j) BLAS-1 loop of
            # modified Gram-Schmidt, and the second pass is taken only when the
            # first one lost norm -- the Daniel-Gragg-Kaufman-Stewart test,
            # which is what "twice is enough" actually licenses.
            #
            # This matters at production size and not at all in a unit test.
            # The orthogonalisation moves O(m^2) full state vectors while the
            # operator applications move O(m); measured at 2048x1024 with
            # m = 8, an unconditional second pass was 53.6% of the substep,
            # and the share grows linearly in m.  Cancellation is rare here,
            # so the test almost always skips it.
            Q = Vflat[: j + 1]
            nrm0 = float(np.sqrt(w @ w))
            hj = Q @ w
            w -= Q.T @ hj
            H[: j + 1, j] += hj
            hnext = float(np.sqrt(w @ w))
            if hnext < _DGKS_ETA * nrm0:
                nreortho += 1
                hj = Q @ w
                w -= Q.T @ hj
                H[: j + 1, j] += hj
                hnext = float(np.sqrt(w @ w))
            H[j + 1, j] = hnext
            if hnext <= 1e-14 * max(1.0, abs(H[: j + 1, j]).max()):
                k_eff = j + 1
                nbrk += 1
                break
            np.multiply(w, 1.0 / hnext, out=w)

        tau = min(tau, t - t_now)
        while True:
            F = _dense_expm(tau * H[:k_eff, :k_eff])
            if k_eff < m:                          # happy breakdown: exact
                err = 0.0
            else:
                err = abs(beta * H[k_eff, k_eff - 1] * F[k_eff - 1, 0])
            if err <= tol * beta or tau <= 1e-12 * float(t):
                break
            nrej += 1
            tau *= max(0.2, 0.9 * (tol * beta / max(err, 1e-300))
                       ** (1.0 / max(k_eff, 1)))

        np.einsum("k,k...->...", beta * F[:k_eff, 0], V[:k_eff], out=v,
                  optimize=True)
        t_now += tau
        nsub += 1
        if err > 0.0:
            grow = 0.9 * (tol * beta / err) ** (1.0 / max(k_eff, 1))
            tau *= min(5.0, max(1.0, grow))

    if stats is not None:
        stats.update(applications=napp, substeps=nsub, rejected=nrej,
                     breakdowns=nbrk, reorthogonalisations=nreortho)
    return v


def krylov_arnoldi_real(kern: "RealHermiteKernel", C: np.ndarray, times,
                        m: int = 8, tol: float = 1e-10, callback=None):
    r"""Arnoldi exponential propagation of the real coefficient field.

    This is the production route required by the specification: the primary
    propagator is the exponential action of the unsplit generator, applied
    matrix-free.  ``callback(index, t, C)`` runs at ``t = 0`` and at each saved
    time.  Returns ``(C_final, stats)``.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("krylov_arnoldi_real: times must be a 1-D array")
    if np.any(np.diff(times) < 0):
        raise ValueError("krylov_arnoldi_real: times must be nondecreasing")

    v = np.ascontiguousarray(C, dtype=np.float64)
    work = np.empty((m + 1,) + v.shape)
    total = {"applications": 0, "substeps": 0, "rejected": 0,
             "breakdowns": 0, "reorthogonalisations": 0}

    if callback is not None:
        callback(0, 0.0, v)

    t_prev = 0.0
    for i, t in enumerate(times):
        dt = float(t) - t_prev
        if dt > 0:
            s = {}
            v = arnoldi_expmv(kern.apply, v, dt, m=m, tol=tol, work=work,
                              stats=s)
            for key in total:
                total[key] += s[key]
        t_prev = float(t)
        if callback is not None:
            callback(i + 1, float(t), v)
    return v, total

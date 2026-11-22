# picturing-fb-reals

Code for the paper *A Fatou–Bieberbach Definition of the Reals That Does Not
Come from Exponential Quotients* by Matthew Frank (on arXiv; the
identifier will be added here).
It produces every figure in the paper and recomputes every numerical
claim in it.

The paper studies the unique entire function f on C^2 with

    f(x, y) = f(4x, 4y)/4 + f(2y, 2x)^2,   f(0,0) = 0,   Df(0,0) = (1,0),

and the set S = {f(x, y) : f(2y, 2x) = 0}, which satisfies
D(0, 3/4) ⊆ S ⊆ D(0, 5/4).

## Notation (as in the paper)

    M(x, y) = (y/2, x/2)  (7),         M^2 = 1/4
    T(x, y) = (x, y + 2x^2),           P = M o T:
    P(x, y) = (x^2 + y/2, x/2),        P^{-1}(x, y) = (2y, 2x - 8y^2)
    B = basin of 0 under P,            K = lim M^{-n} P^n on B
    H = (f, g) = K^{-1},               g(z) = f(M^{-1} z)/2
    V_g = {g = 0} = H^{-1}(C x {0}),   f: V_g -> S biholomorphic,
                                       inverse w -> K(w, 0)
    V_f = M^{-1} V_g = {f = 0},        q = (3/4, 3/8), a saddle of P
    K^-, K^+ = points with bounded backward, forward orbits under P
    G^- = backward Green function,     A(t) = G^-(H(t, 0))
    Q^s = H o 4^{-s} o K (the flow)

Sets, written in bold in the paper, are plain capitals here.  The
paper's table of notation gives the general conventions.

The figures label the zero of f over w in S by (u, v) = M^{-1} K(w, 0),
so that f(u, v) = 0 and f(v/2, u/2) = w.  Everything in the code uses
this notation; there is no internal rescaling.

## Reproducing everything

With Python 3.13 (other recent versions should work):

    pip install -r requirements.txt
    python make_fig_contour.py     # Figure 1                     ~25 s
    python make_fig_S.py           # Figure 2                     ~10 s
    python make_fig_R.py           # Figure 3                      ~5 s
    python make_fig_growth.py      # Figure 4 (Sec. 6.1)          ~15 s
    python verify_growth.py        # certified, Sec. 6.1           ~1 s
    python claims.py               # every numerical claim        ~2 min
                                   #   (+2 min the first time, to
                                   #   fill cache/edges.npy)
    python -m pytest               # identities, as tests          ~1 s

The figures are written to `figures/` as `.pdf` (used in the paper) and
`.png`, with a date suffix set by `FIG_SUFFIX` in `common.py`, e.g.
`figures/fig_S_1119.pdf`; `--out FILE.pdf` writes elsewhere.  Outputs
are deterministic: rerunning a script on the same platform gives
byte-identical files with the pinned library versions; elsewhere, fonts
and system libraries may change the bytes but not the figures.  No script uses absolute paths; each can
be run from any directory.

## Certified computations

The proofs in the paper use no computation.  One certified computation
illustrates Section 6.1, in outward-rounded interval arithmetic
(mpmath's `iv` context, mpmath 1.4.1); it is not needed for any proof,
since Theorem 6.3 is proved there without computation:

- `python verify_growth.py` prints certified enclosures
  [1.6912613, 1.7031209] and [2.9661488, 2.9714061] of A(t)/t on
  [1.37, 1.38] and [2.49, 2.51], showing that the escape rate
  A(t) = G^-(H(t, 0)) is far from homogeneous.  The script is
  self-contained, and its docstring states each step.  It is also run,
  on two short pieces, by the test `test_growth_on_axis_not_regular`.

Everything else in `claims.py`, and every figure, uses floating-point
or high-precision arithmetic without rigorous error bounds, and is
numerical evidence only.  Three sources of error matter: truncation of
the Taylor series of H near 0, which is negligible at the radius used;
loss of precision under backward iteration where f is small compared
with the intermediate values, as near its zeros at large |t|; and
points near the boundary of S or of the basin, which may neither
converge nor escape within the iteration limit and are then counted as
outside.

## Which script produces what

| Paper | Script |
|---|---|
| Figure 1 (contours of f; the real zeros and the points (v/2, u/2)) | `make_fig_contour.py` |
| Figure 2 (S with the curves \|u\| = 4^k and the zeros of u) | `make_fig_S.py` |
| (not in the paper) S with \|v\| = 4^k and the zeros of v | `make_fig_S.py --modulus v` |
| Figure 3 (the sets {v : exists m, nv - m in S}, n = 2, 4) | `make_fig_R.py` |
| Figure 4, Sec. 6.1 (H(t, 0) with level curves of G^- and K^- in the real plane; A(t)/\|t\| and T(r)/r, T the Nevanlinna characteristic of f(t, 0)) | `make_fig_growth.py` |
| Taylor coefficients (8) | `claims.py`, tests |
| Fig. 1, Sec. 2, Sec. 4.3: real segment (-0.9903, 0.8566) of S | `claims.py` |
| Sec. 2: max of f on [-3,3]^2 (about 1.85); sup of f on R^2 (3.44401) | `claims.py` |
| Fig. 2: 0.857 <= \|w\| <= 1.063 on the boundary of S; Sec. 4.4: rays leave S once | `claims.py` |
| Fig. 1, Sec. 4.2: amplitude about 1.3e-4 of y/x - 1 on the real edges | `claims.py` |
| Sec. 5.2: v/u takes essentially all values on the zero curve V | `claims.py` |
| Sec. 6.1: certified G^-(H(t, 0))/t on two intervals; its range over a period | `verify_growth.py`, `claims.py`, tests |
| Sec. 4.2: about twelve digits for \|z\| <= 10; spot checks of (2), (20), det DH = 1 | `claims.py` |
| Sec. 6.1, Figure 4: Nevanlinna characteristic T(r) of f(t, 0): T(4r) - 4T(r) - 3 log 4 and T(r) + log 4 - (mean of A)/2 below 1e-4 for r >= 6; (T(r) + log 4)/r from about 1.355 to 1.685 | `claims.py` |
| Not in paper (cf. Sec. 6.5): test function f_E(t) = t E_{3,2}(1 - 8t^3)/E_{3,2}(1): T_fE(r)/r -> 3 sqrt 3/pi; ratio T_fE/T_f apparently eventually between 0.98 and 1.22 | `claims.py` |
| (2), P o H = H o M, K o H = id, (11), Q^-1(z/4) = T_0^-1 T_1^-1 z, symmetry, bounds on S | `tests/` |

`claims.py` prints each value computed next to the value stated in the
paper, with the section of the paper where it appears.  Claims
labelled `Not in paper` are further computations, not stated in the
current version of the paper; their reference values are printed as
`expected`.

## Files

- `fnum.py`: the mathematics, with functions named as in the paper:
  the maps `P`, `P_inv`, `M`, `M_inv`, `T`, `T_k_inv`, `sigma`;
  `taylor_coefficients` (recursion (11), exact or floating);
  `H`, `f`, `g` in double precision (Taylor series of degree 40 near 0
  with backward iteration, about 12 digits for |z| up to about 10);
  `H_mp`, `f_mp` in multiprecision (the shear product (18)–(19));
  the forward route `K`, `K_w`, `K_w_mp`, `uv_from_w` and the
  vectorized escape test `K_w_grid`, `in_S`; the basin test `in_basin`,
  `unstable_branch` and the backward Green function `G_minus`.  It also
  contains functions used only by the companion paper on the real
  dynamics (the flow lines `flow_line`, and Newton continuation of the
  curves in S on which u is real), kept so that its code can depend on
  a fixed release of this one.
- `common.py`: paths, deterministic saving of figures, and the caches.
- `make_fig_*.py`: one script per figure.
- `claims.py`: every numerical claim of the paper.
- `verify_growth.py`: certified escape rates (Sec. 6.1), without `fnum`.
- `tests/test_identities.py`: pytest tests of the exact identities.

## Cache

`claims.py` uses one array in `cache/`, saved as a plain `.npy` file
(loadable with `allow_pickle=False`).  It is recomputed automatically
when missing, or with `--recompute`.  It is not part of the repository
(see `.gitignore`).

| File | Contents | Time to recompute |
|---|---|---|
| `edges.npy` | rows (side, L, y/x - 1) on the real edges of {f > 0} at x = 4^L, 0 <= L <= 8, step 1/32 (60 digits) | ~2 min |

## Companion paper

The real dynamics pictured in the companion paper *Picturing the real
dynamics of a Fatou–Bieberbach map* (in preparation) are computed in a
separate repository, which depends on a fixed release of this one.

## Dependencies and their licenses

| Package | License |
|---|---|
| NumPy | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) |
| SciPy | BSD-3-Clause |
| Matplotlib | Matplotlib License (PSF-based, BSD-compatible) |
| mpmath | BSD-3-Clause |
| pytest (tests only) | MIT |

All are permissive licenses compatible with the MIT License of this
code, as are the libraries they in turn depend on.  None of them is
included in this repository.

## License

The code is licensed under the MIT License; see `LICENSE`.

This repository ships no data files: the cached arrays above are
recomputed locally by the scripts.  Any data files distributed with a
release of this code are licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).

The paper itself is not part of this repository; it is available on
arXiv under CC BY 4.0, a separate license from that of the code.

## Citation

Please cite the paper; see `CITATION.cff` (GitHub shows it under "Cite
this repository").  Releases are archived on Zenodo.

## Acknowledgement

This code was written and developed with Claude and ChatGPT.

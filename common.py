"""Paths, figure saving, and the caches.

Caches (plain .npy arrays, loadable with allow_pickle=False) live in
cache/ and are recomputed when missing or when --recompute is given.
This paper uses only cache/edges.npy; the two sheet caches serve the
companion paper's repository, which depends on a release of this one:

  cache/edges.npy      the real edges of {f > 0}: rows (side, L, xi),
                       side = +1 (upper edge, w > 0) or -1 (lower edge),
                       x = 4^L, xi = y/x - 1 at the zero (x, y) of f
  cache/sheets_pos.npy curves in S on which u is real, u > 0,
  cache/sheets_neg.npy and u < 0: rows (curve, level, Re w, Im w); the
                       first row of each curve is the zero of u it
                       starts from
"""
import argparse
from pathlib import Path
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import fnum

ROOT = Path(__file__).resolve().parent
CACHE = ROOT/'cache'
FIGURES = ROOT/'figures'
FIG_SUFFIX = '_1119'          # date suffix of the figures in the paper

W_LO = '-0.99032708742948748318'      # ends of S on the real axis
W_HI = '0.85656537346597551055'
GRID_L, GRID_N = 1.08, 1600           # grid for S in Figures 2-4

# Lighter bold for the bold letters (Omega, W, S) in the legends of
# Figures 5 and 7: Inter SemiBold rather than DejaVu Sans Bold, with the
# other math glyphs unchanged.  Falls back to DejaVu Sans if Inter is
# not installed.
LIGHT_BOLD = {'mathtext.fontset': 'custom',
              'mathtext.rm': 'DejaVu Sans',
              'mathtext.it': 'DejaVu Sans:italic',
              'mathtext.sf': 'DejaVu Sans',
              'mathtext.cal': 'DejaVu Sans',
              'mathtext.bf': 'Inter:weight=600'}


def figure_path(stem, out=None):
    """figures/<stem><FIG_SUFFIX>.pdf unless out is given."""
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        return Path(out)
    FIGURES.mkdir(exist_ok=True)
    return FIGURES/f'{stem}{FIG_SUFFIX}.pdf'


def savefig(fig, path, dpi):
    """Save as .pdf and .png without timestamps (deterministic)."""
    path = Path(path)
    fig.savefig(path, bbox_inches='tight',
                metadata={'CreationDate': None})
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')


def parser(doc, recompute=False):
    p = argparse.ArgumentParser(description=doc.splitlines()[0])
    p.add_argument('--out', help='output .pdf (a .png is written too)')
    if recompute:
        p.add_argument('--recompute', action='store_true',
                       help='recompute the cached curves in cache/')
    return p


# ------------------------------------------------------------------
# The real edges of {f > 0}: for real w in S, (u, v) = M^{-1} K(w, 0)
# is a real zero of f, and u increases from 0 to +inf as w runs from 0
# to either end of the real segment of S.  Each x = 4^L,
# 0 <= L <= 8, step 1/32, is hit by bisection on w at 60 digits
# (190 steps).  About 3 minutes.
# ------------------------------------------------------------------
def edge_data(recompute=False, step=1/32):
    path = CACHE/'edges.npy'
    if path.exists() and not recompute:
        return np.load(path, allow_pickle=False)
    rows = []
    with mp.workdps(60):
        for side, end in ((-1, mp.mpf(W_LO)), (1, mp.mpf(W_HI))):
            start = side*mp.mpf('1e-20')
            for L in np.arange(0.0, 8.0 + 1e-9, step):
                target = mp.mpf(4)**L
                lo, hi = start, end          # u increases toward end
                for _ in range(190):
                    mid = (lo + hi)/2
                    if fnum.uv_from_w(mid)[0] < target:
                        lo = mid
                    else:
                        hi = mid
                u, v = fnum.uv_from_w((lo + hi)/2)
                rows.append((side, L, float(v/u - 1)))
    D = np.array(rows)
    CACHE.mkdir(exist_ok=True)
    np.save(path, D)
    return D


# ------------------------------------------------------------------
# The curves in S on which u is real, from the zeros of u.  Levels of
# zeros: 0 (the double zero w = 0), 1 (six zeros, |v| = 2.78), 2 (24
# zeros, |v| about 11), 3 (96 zeros).  Sampled every factor 1.03 in |u|
# up to 1e15.  About 2 minutes for both signs.
# ------------------------------------------------------------------
def s_grid():
    """S on the GRID_N^2 grid of [-GRID_L, GRID_L]^2 (rows from the top):
    (xs, inS, U, V, log2|u|, log2|v|)."""
    xs = np.linspace(-GRID_L, GRID_L, GRID_N)
    return (xs,) + fnum.K_w_grid(xs[None, :] + 1j*xs[::-1, None])


def zeros_of_u(xs=None, inS=None, Lu=None):
    """The zeros of u found on the grid, with v there."""
    if xs is None:
        xs, inS, _, _, Lu, _ = s_grid()
    Z = fnum.zeros_on_grid(xs, inS, Lu)
    V = np.array([fnum.uv_from_w(z)[1] for z in Z])
    Z = np.where(abs(Z) < 1e-5, 0j, Z)        # the double zero at 0
    return Z, V


def level(z0, v0):
    return 0 if abs(z0) < 1e-9 else (1 if abs(v0) < 5 else
                                     2 if abs(v0) < 20 else 3)


def sheet_data(sign, Z=None, V=None, recompute=False):
    """List of (level, z0, path) for the curves with sign(u) = sign."""
    path = CACHE/('sheets_pos.npy' if sign > 0 else 'sheets_neg.npy')
    if path.exists() and not recompute:
        return unpack_sheets(np.load(path, allow_pickle=False))
    if Z is None:
        Z, V = zeros_of_u()
    out = []
    for z0, v0 in zip(Z, V):
        lev = level(z0, v0)
        if sign < 0 and lev == 3:
            continue
        for s, p in fnum.branches(z0, double=(lev == 0),
                                  eps=(1e-3, 1e-3, 2e-4, 5e-5)[lev],
                                  factor=1.03):
            if s == sign:
                out.append((lev, z0, p))
    CACHE.mkdir(exist_ok=True)
    np.save(path, pack_sheets(out))
    return out


def pack_sheets(sheets):
    rows = [np.column_stack([np.full(len(p), i), np.full(len(p), lev),
                             p.real, p.imag])
            for i, (lev, z0, p) in enumerate(sheets)]
    return np.vstack(rows)


def unpack_sheets(A):
    out = []
    for i in np.unique(A[:, 0]):
        B = A[A[:, 0] == i]
        p = B[:, 2] + 1j*B[:, 3]
        out.append((int(B[0, 1]), p[0], p))
    return out

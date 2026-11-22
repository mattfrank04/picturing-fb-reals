"""Figure 3: defining R from f and the integers.

v is real iff for every integer n there is an integer m with
nv - m in S.  Panel n shows {v : exists m, nv - m in S}, the union of
the copies (S + m)/n, m in Z: copies of S/2 at the half-integers
(n = 2), then copies of S/4 at the quarter-integers (n = 4).  Every
panel contains R, and a non-real v drops out of panel n once
|n Im v| >= 5/4.

S = {w : (w, 0) in B} is computed by escape testing (fnum.in_S, as
for Figure 2).  About 5 s.

    python make_fig_R.py [--out FILE.pdf]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fnum
import common

BLUE, FILL = '#1f4e9c', '#c9d7ee'
L, N = 1.1, 1200                  # grid for S: [-L, L]^2, N x N
X0, X1 = -2.125, 2.125              # visible range of Re v
NS = (2, 4)                       # one panel per n


def s_mask(xs):
    """inS on the grid xs + i xs (rows from the top), as in Figure 2."""
    return fnum.in_S(xs[None, :] + 1j*xs[::-1, None]).astype(bool)


def lookup(mask, xs, z):
    """Nearest-grid value of the S mask at the points z."""
    n = len(xs)
    j = np.round((z.real + L)/(2*L)*(n - 1)).astype(int)
    i = np.round((L - z.imag)/(2*L)*(n - 1)).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    out = np.zeros(z.shape, bool)
    out[ok] = mask[i[ok], j[ok]]
    return out


def row_mask(mask, xs, V, n):
    """{v : exists m in Z, nv - m in S} on the plot grid V."""
    z = n*V
    m0 = np.floor(z.real)
    out = np.zeros(V.shape, bool)
    for dm in range(-2, 4):           # S lies in |w| < 5/4
        out |= lookup(mask, xs, z - (m0 + dm))
    return out


def main(out=None):
    out = common.figure_path('fig_R', out)
    xs = np.linspace(-L, L, N)
    mask = s_mask(xs)
    ylim = 1.0
    px = np.linspace(X0, X1, 2400)
    py = np.linspace(-ylim, ylim, 420)
    V = px[None, :] + 1j*py[:, None]

    fig, axs = plt.subplots(1, len(NS), figsize=(10, 2.6),
                            sharey=True, gridspec_kw=dict(wspace=.06))
    for a, n in zip(np.atleast_1d(axs), NS):
        a.contourf(px, py, row_mask(mask, xs, V, n).astype(float),
                   levels=[.5, 1.5], colors=[FILL])
        for m in range(int(np.floor(n*X0)) - 2, int(np.ceil(n*X1)) + 3):
            a.contour((xs + m)/n, xs[::-1]/n, mask.astype(float),
                      levels=[.5], colors=BLUE, linewidths=.5)
        a.axhline(0, color='k', lw=.9)
        a.set(xlim=(X0, X1), ylim=(-ylim, ylim), aspect='equal')
        a.set_yticks([-1, 0, 1])
        a.set_xticks([-2, -1, 0, 1, 2])
        a.set_xlabel('Re $x$')
        lab = (r'$\{x:\exists m\ x-m\in \mathbf{S}\}$' if n == 1 else
               r'$\{x:\exists m\ %dx-m\in \mathbf{S}\}$' % n)
        a.set_title(lab, fontsize=10)
    np.atleast_1d(axs)[0].set_ylabel('Im $x$')
    common.savefig(fig, out, dpi=150)


if __name__ == '__main__':
    main(common.parser(__doc__).parse_args().out)

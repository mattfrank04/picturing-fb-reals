"""Figure 2: S = {f(v/2,u/2) : f(u,v) = 0}, with the curves |u| = 4^k
and the zeros of u, where (u, v) = M^{-1} K(w, 0) is the zero of f
over w in S.  With --modulus v it draws |v| = 4^k and the zeros of v
instead (not used in the paper).

S, u and v come from the forward route (fnum.K_w_grid) by escape testing
on a 1600^2 grid; zeros are refined by secant steps on w -> u(w) or
v(w).  About 10 s.

    python make_fig_S.py [--modulus u|v] [--out FILE.pdf]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import fnum
import common

L, N = common.GRID_L, common.GRID_N


def circles(a):
    for r in (0.75, 1.25):
        a.add_patch(Circle((0, 0), r, fill=False, ls='--', lw=.7,
                           color='0.7'))


def main(out=None, modulus='u'):
    c = 'uv'.index(modulus)                    # 0 for u, 1 for v
    out = common.figure_path('fig_S' if c == 0 else 'fig_S_v', out)
    xs = np.linspace(-L, L, N)
    W = xs[None, :] + 1j*xs[::-1, None]
    inS, U, V, Lu, Lv = fnum.K_w_grid(W)
    L2 = (Lu, Lv)[c]
    Z = fnum.zeros_on_grid(xs, inS, L2, comp=c)
    other = np.array([fnum.uv_from_w(z)[1 - c] for z in Z])
    Z = np.where(abs(Z) < 1e-5, 0j, Z)        # the zero at 0

    fig, a = plt.subplots(figsize=(9, 8.6))
    lim = dict(xlim=(-1.52, 1.48), ylim=(-1.32, 1.32))
    bd = dict(levels=[.5], colors='k', linewidths=.8)
    ii = lambda v: np.clip(np.round((L - v.imag)/(2*L)*(N - 1))
                           .astype(int), 0, N - 1)
    jj = lambda v: np.clip(np.round((v.real + L)/(2*L)*(N - 1))
                           .astype(int), 0, N - 1)

    # |u| = 4^k = 1/16, ..., 256, or |v| = 4^k = 1/16, ..., 1024
    lv = np.arange(-4, 10 if c == 0 else 12, 2)
    cs = a.contour(xs, xs[::-1], L2, levels=lv, cmap='viridis',
                   linewidths=.8)
    a.contour(xs, xs[::-1], inS.astype(float), **bd)
    circles(a)
    ray = (np.linspace(0.01, 1.0, 4000) if c == 0 else
           np.linspace(0.01, 1.1, 6000))*np.exp(1j*np.deg2rad(205))
    vals = L2[ii(ray), jj(ray)]
    pos = []
    for l in ([-4, -2, 0, 2, 4, 6] if c == 0 else [-4, -2, 0, 2, 4]):
        k = np.where(np.isfinite(vals[:-1]) & (vals[:-1] < l)
                     & (vals[1:] >= l))[0]
        if len(k):
            pos.append((l, ray[k[0]].real, ray[k[0]].imag))
    if c == 0:              # one more label, near a zero of u, |v| = 2.78
        inner = Z[abs(other) < 4]
        z0 = inner[np.argmin(abs(inner - (-0.7448 - 0.2711j)))]
        ray2 = z0 + np.linspace(0.002, 0.12, 2000)*np.exp(
            1j*np.deg2rad(225))
        v2 = L2[ii(ray2), jj(ray2)]
        k2 = np.where((v2[:-1] < 0) & (v2[1:] >= 0))[0]
        if len(k2):
            pos.append((0, ray2[k2[0]].real, ray2[k2[0]].imag))
    fmt = {l: (f'{2**l:g}' if l >= 0 else f'1/{2**(-l):g}') for l in lv}
    a.clabel(cs, fmt=fmt, fontsize=7, inline=True, inline_spacing=2,
             manual=[(x, y) for l, x, y in pos])
    top = ('64, 256', '256, 1024')[c]
    a.text(0.02, 0.02, f'contours: |{modulus}| = 4^k = 1/16, 1/4, … , {top}',
           transform=a.transAxes, fontsize=7.5, color='0.3')
    a.plot(Z.real, Z.imag, 'o', ms=1.6, color='0.2', zorder=4)
    a.text(0.02, 0.09, f'dots: zeros of {modulus}',
           transform=a.transAxes, fontsize=7.5, color='0.3')
    a.text(0.02, 0.055, 'dashes: |w| = 3/4 and 5/4',
           transform=a.transAxes, fontsize=7.5, color='0.3')
    a.set_title(r'$\mathbf{S}=\{f(z) : f(4z)=4f(z)\}$, with curves where '
                r'$|%s| = 4^k$' % modulus)
    a.set(**lim)
    a.set_aspect('equal')
    a.set_xlabel('Re $w$')
    a.set_ylabel('Im $w$')

    common.savefig(fig, out, dpi=130)


if __name__ == '__main__':
    p = common.parser(__doc__)
    p.add_argument('--modulus', choices=('u', 'v'), default='u',
                   help='draw |u| = 4^k (Figure 2) or |v| = 4^k')
    args = p.parse_args()
    main(args.out, args.modulus)

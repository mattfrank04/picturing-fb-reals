"""Figure 1: contours of f on [-3,3]^2 (top); on [-1,2]^2, the contours
f = +-1, the real zero curve of f, and the points (v/2, u/2) for its
zeros (u, v) = M^{-1} K(w, 0), w real in S (bottom).

Contours: double-precision H (Taylor series with backward iteration).
Zeros: the forward route, no root-finding in f.  About 25 s.

    python make_fig_contour.py [--out FILE.pdf]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import fnum
import common

RED, BLUE, FILL = '#c0392b', '#2c5fa8', '#fde9c9'
BAND = '#e4ecf5'
PURPLE = '#7b3fa0'
W_LO, W_HI = common.W_LO, common.W_HI     # ends of S on R


def zero_curve(ax, X, Y, F):
    """Real zero curve of f: upper edge (w > 0) red, lower (w < 0) blue."""
    zc = plt.figure().add_subplot().contour(X, Y, F, levels=[0])
    segs, cols = [], []
    for p in zc.allsegs[0]:
        if len(p) < 2:
            continue
        w = fnum.f(p[:, 1]/2, p[:, 0]/2).real    # w = f(y/2, x/2)
        for k in range(len(p) - 1):
            segs.append(p[k:k + 2])
            cols.append(RED if w[k] + w[k + 1] > 0 else BLUE)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=2,
                                     zorder=4))


FAR = {-10: (-2.35, 0.2), -100: (-2.45, -1.15), -1000: (-2.7, -2.3)}


def main(out=None):
    out = common.figure_path('fig_contour', out)
    x0, x1, n = -3.0, 3.0, 900
    xs = np.linspace(x0, x1, n)
    X, Y = np.meshgrid(xs, xs)
    F = fnum.f(X, Y).real                          # the paper's f

    fig = plt.figure(figsize=(7.6, 12.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 2.25], hspace=0.17)
    ax = fig.add_subplot(gs[0])
    levels = [-1000, -100, -10, -4, -3, -2, -1, 1]
    cs = ax.contour(X, Y, F, levels=levels, colors='0.55', linewidths=.7,
                    linestyles='solid')
    # labels: the negative levels where they cross x = 1.15 below the
    # zero curve and where they cross the segment from (0,1) to (2,3);
    # the level 1 inside the strip
    col = F[:, np.argmin(abs(xs - 1.15))]
    seg = np.linspace(0, 1, 2000)
    sx, sy = 2*seg, 1 + 2*seg
    fseg = fnum.f(sx, sy).real
    for lev in levels:
        pos = []
        if lev in FAR:
            pos.append(FAR[lev])              # on the outer arcs
        elif lev < 0:
            k = np.where((xs < 0) & (np.diff(np.sign(col - lev),
                                             append=0) != 0))[0]
            pos.append((1.15, xs[k[-1]]))
            k = np.where(np.diff(np.sign(fseg - lev)) != 0)[0]
            if len(k):
                pos.append((sx[k[0]], sy[k[0]]))
        else:
            pos.append((1.45, 0.8))
        ax.clabel(cs, [lev], fmt='%+d' if lev > 0 else '%d', fontsize=8,
                  inline=True,
                  manual=pos)
    zero_curve(ax, X, Y, F)
    ax.plot([x0, x1], [x0, x1], '--', color='0.35', lw=.7)
    ax.add_patch(plt.Rectangle((-1, -1), 3, 3, fill=False, ec='0.3',
                               lw=.8, ls=':'))
    ax.plot([], [], color=RED, lw=2, label='$f=0$, upper edge')
    ax.plot([], [], color=BLUE, lw=2, label='$f=0$, lower edge')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.set(xlim=(x0, x1), ylim=(x0, x1), aspect='equal')
    ax.set_xlabel('$x$', labelpad=1)
    ax.set_ylabel('$y$')
    ax.set_title('Integer contours of $f$ on $[-3,3]^2$', fontsize=11)

    # bottom: [-1,2]^2, the zeros (u, v) and the points (v/2, u/2)
    b = fig.add_subplot(gs[1])
    a0, a1 = -1.0, 2.0
    ys = np.linspace(a0 - .05, a1 + .05, 700)
    Xb, Yb = np.meshgrid(ys, ys)
    Fb = fnum.f(Xb, Yb).real
    cb = b.contour(Xb, Yb, Fb, levels=[-1, 1], colors='0.55',
                   linewidths=.7, linestyles='solid')
    b.clabel(cb, [-1], fmt='%d', fontsize=8, inline=True,
             manual=[(1.6, -0.3)])
    b.clabel(cb, [1], fmt='%+d', fontsize=8, inline=True,
             manual=[(1.2, 0.75)])
    zero_curve(b, Xb, Yb, Fb)
    for end in (float(W_LO), float(W_HI)):
        w = np.concatenate([np.linspace(0, end, 400)[:-1],
                            end - np.sign(end)*np.logspace(-2.5, -15, 300)])
        uv = np.array([fnum.uv_from_w(t) for t in w])
        px, py = uv[:, 1].real/2, uv[:, 0].real/2
        keep = (px < a1 + .2) & (py < a1 + .2)
        b.plot(px[keep], py[keep], color=RED if end > 0 else BLUE,
               lw=1.3, ls=(0, (4, 2.5)), zorder=5)
    b.plot([], [], color='0.3', lw=2, label='$f(u,v)=0$')
    b.plot([], [], color='0.3', lw=1.3, ls=(0, (4, 2.5)),
           label='$(v/2,\\,u/2)$ with $f(u,v)=0$')
    b.legend(loc='lower right', fontsize=9, frameon=False)
    b.set(xlim=(a0, a1), ylim=(a0, a1), aspect='equal')
    b.set_xlabel('$x$', labelpad=1)
    b.set_ylabel('$y$')
    b.set_title('Contours $f=\\pm1$ and $f=0$ on $[-1,2]^2$, and the '
                'points $(v/2,u/2)$', fontsize=11)
    common.savefig(fig, out, dpi=140)


if __name__ == '__main__':
    main(common.parser(__doc__).parse_args().out)

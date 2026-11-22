"""Figure 4 (Section 6.1): growth of f along the real axis and on circles.

Left: the paths H(t, 0) for t > 0 (gold) and t < 0 (brown), each to the
edge of the window, in the real plane, with the boundary of the real
basin Omega, the real points K^- of R^2 with bounded backward orbits,
through the saddle q = (3/4, 3/8) (dark purple), and level curves
(green) of the backward Green function G^- = 1/3, 2/3, ..., 3, light
to dark.  Right, against log_4|t|: A(t)/|t| for t < 0 (brown) and
t > 0 (gold), where
A(t) = G^-(H(t, 0)), and T(|t|)/|t| (black) for the Nevanlinna
characteristic T of t -> f(t, 0), dashed.  One legend under each panel.

The layout constants below (figure size, axes boxes in figure
fractions, and the horizontal extent of the crop in inches) match the
companion paper's flow-line figure.  Double precision; numerical
illustration only.

    python make_fig_growth.py [--out FILE.pdf]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import Bbox
import fnum
import common

matplotlib.rcParams.update(common.LIGHT_BOLD)

FIGSIZE = (6.5, 3.75)                       # fixed layout
AX_LEFT = (0.0968436, 0.1853134, 0.3761375, 0.6193731)
AX_RIGHT = (0.6022245, 0.1853134, 0.2977755, 0.6193731)
X_EXTENT = (0.2128223, 6.2246832)           # inches, crop
GOLD, BROWN = '#c08a00', '#8c3d10'   # t > 0, t < 0
PURPLE = '#5b2c83'                  # K^- (dark purple)
DOT = 1.9                           # 0.6 of the former marker size 3
S_MAX = 3.5                         # right panel: 0 <= log_4 r <= 3.5
N_PHI = 2000                        # angles in [0, pi/3] for T(r)


def characteristic(r):
    """Nevanlinna characteristic T(r) = m(r) of the entire function
    t -> f(t, 0): the mean of log+|f(r e^{i phi}, 0)| over the circle.
    As |f(omega t, 0)| = |f(t, 0)| and f is real, the mean over
    [0, pi/3] (midpoint rule) is the mean over the circle.  Double
    precision suffices up to r = 4^3.5 (log|f| < 709 there)."""
    phi = (np.arange(N_PHI) + 0.5)*(np.pi/3)/N_PHI
    t = r*np.exp(1j*phi)
    X, _ = fnum.H(t, 0*t)
    return float(np.mean(np.log(np.maximum(np.abs(X), 1.0))))
LEV = np.arange(1, 10)/3
GM = 'G^{\\text{–}}'                    # G with an en-dash


def main(out=None):
    out = common.figure_path('fig_growth', out)
    cm = matplotlib.colormaps['Greens']
    cols = [cm(0.35 + 0.6*i/(len(LEV) - 1)) for i in range(len(LEV))]
    xs = np.linspace(-1.2, 1.2, 700)
    ys = np.linspace(-2.6, 2.6, 1500)
    inside = fnum.in_basin(xs[None, :], ys[:, None])
    G = fnum.G_minus(xs[None, :], ys[:, None])
    # each path runs to the edge of the window
    tp = np.linspace(0, 1.5256, 1900)
    tn = np.linspace(-0.8249, 0, 1700)
    Hp, Hn = fnum.H(tp, 0*tp), fnum.H(tn, 0*tn)
    # A(t)/|t| has period 1 in log_4|t| exactly, as A(4t) = 4A(t):
    # compute one period and repeat it across the window.
    u = np.linspace(0, 1, 1001)[:-1]
    s_win = np.linspace(0, S_MAX, 3501)

    def ratio(s):
        X, Y = fnum.H(s, 0*s)
        return fnum.G_minus(X.real, Y.real)/np.abs(s)

    def periodic(vals):
        return np.interp(s_win % 1, np.r_[u, 1], np.r_[vals, vals[0]])

    Ap, An = periodic(ratio(4.0**u)), periodic(ratio(-4.0**u))
    sT = np.linspace(0, S_MAX, 561)
    TT = np.array([characteristic(4.0**k) for k in sT])/4.0**sT

    fig = plt.figure(figsize=FIGSIZE)
    a, b = fig.add_axes(AX_LEFT), fig.add_axes(AX_RIGHT)
    a.contour(xs, ys, G, levels=LEV, colors=cols, linewidths=.6)
    a.contour(xs, ys, inside.astype(float), levels=[.5], colors='k',
              linewidths=.8)
    a.plot(Hp[0].real, Hp[1].real, color=GOLD, lw=1.1)
    a.plot(Hn[0].real, Hn[1].real, color=BROWN, lw=1.1)
    for side in (-1, 1):                    # U into 0, U' outward
        ub = fnum.unstable_branch(steps=60 if side < 0 else 34,
                                  side=side)
        a.plot(ub[:, 0], ub[:, 1], color=PURPLE, lw=.8, zorder=4)
    a.plot(.75, .375, 'o', ms=DOT, color=PURPLE, zorder=6)    # q
    a.set(xlim=(-1.2, 1.2), ylim=(-2.6, 2.6))
    a.set_xlabel('$x$', labelpad=1)
    a.set_ylabel('$y$', labelpad=1)
    a.set_title(f'$H(t,0)$ and level curves of ${GM}$', fontsize=9.5)
    b.plot(s_win, Ap, color=GOLD, lw=1.1)
    b.plot(s_win, An, color=BROWN, lw=1.1)
    b.plot(sT, TT, color='k', lw=1.1, ls=(0, (4, 2)))
    b.set_xlim(0, S_MAX)
    b.set_ylim(0, 4)
    b.set_yticks([0, 1, 2, 3, 4])
    b.set_xticks([0, 1, 2, 3])
    b.set_xlabel('$\\log_4|t|$', labelpad=1)
    b.set_title('Growth along rays and circles', fontsize=9.5)
    for ax in (a, b):
        ax.tick_params(labelsize=8)
    om = '\\boldsymbol{\\Omega}'
    kw = dict(loc='upper left', fontsize=8.5, frameon=False,
              labelspacing=0.25, handletextpad=0.6, borderaxespad=0.1)
    # left legend, under the left panel: the plane
    hl = [Line2D([], [], color='k', lw=.8),
          Line2D([], [], color=PURPLE, lw=.8, marker='o', ms=DOT),
          tuple(Line2D([], [], color=cols[i], lw=1.2) for i in (0, 4, 8))]
    ll = [f'boundary of ${om}$',
          '$\\mathbf{K}^-\\cap\\mathbb{R}^2$, through $q$',
          f'${GM}=\\frac{{1}}{{3}},\\frac{{2}}{{3}},\\ldots,3$ '
          '(light to dark)']
    fig.legend(hl, ll, handler_map={tuple: HandlerTuple(ndivide=None,
                                                         pad=0)},
               bbox_to_anchor=(AX_LEFT[0] - 0.03, 0.085), **kw)
    # right legend, under the right panel: the paths and the growth
    hr = [Line2D([], [], color=BROWN, lw=1.1),
          Line2D([], [], color=GOLD, lw=1.1),
          Line2D([], [], color='k', lw=1.1, ls=(0, (4, 2)))]
    lr = ['$H(t,0)$ and $A(t)/|t|$, $t<0$',
          '$H(t,0)$ and $A(t)/|t|$, $t>0$',
          'Nevanlinna characteristic $T(|t|)/|t|$']
    fig.legend(hr, lr, bbox_to_anchor=(AX_RIGHT[0] - 0.03, 0.085), **kw)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    box = Bbox.from_extents(X_EXTENT[0] - .1, bb.y0 - .1,
                            X_EXTENT[1] + .1, bb.y1 + .1)
    fig.savefig(out, bbox_inches=box, metadata={'CreationDate': None})
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches=box)


if __name__ == '__main__':
    args = common.parser(__doc__).parse_args()
    main(args.out)

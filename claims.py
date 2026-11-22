"""Recompute every numerical claim in the paper and print each value
next to the value stated in the paper (about 3 minutes, plus about 4
minutes the first time, to fill cache/).

    python claims.py [--recompute]

Each line reads:  [where in the paper] claim: paper value | computed.
"""
import argparse
import time
from math import factorial
import numpy as np
import mpmath as mp
from scipy import ndimage
import fnum
import common

P, P_inv = fnum.P, fnum.P_inv


def report(where, claim, paper, computed):
    """Print a claim of the paper with its computed value.  Claims
    labelled 'Not in paper' are further computations, not stated in the
    current version; their reference values are printed as 'expected'."""
    label = 'expected' if where.startswith('Not in paper') else 'paper'
    print(f'[{where}] {claim}:\n    {label}: {paper}\n    code:  {computed}')


def ratio_on_V():
    """Sec. 5.2: v/u on the zero curve V takes essentially all values."""
    xs, inS, U, V, Lu, Lv = common.s_grid()
    ok = inS & np.isfinite(U) & (abs(U) > 0)
    r = V[ok]/U[ok]
    L, h = 4, 0.05
    n = int(2*L/h)
    i = np.floor((r.real + L)/h).astype(int)
    j = np.floor((r.imag + L)/h).astype(int)
    m = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    C = np.zeros((n, n), bool)
    C[i[m], j[m]] = True
    inner = C[n//4:3*n//4, n//4:3*n//4].mean()
    report('Sec. 5.2', 'v/u takes essentially all values on V',
           'yes', 'grid samples hit %.1f%% of cells of [-4,4]^2 and %.1f%% '
           'of [-2,2]^2 (cells 0.05); max |v/u| %.0f'
           % (100*C.mean(), 100*inner, abs(r).max()))


def growth_on_axis():
    """Sec. 6.1: G^-(H(t, 0))/t on two intervals of t."""
    import verify_growth
    (lo1, hi1), (lo2, hi2), ok = verify_growth.certify()
    lo1, hi1, lo2, hi2 = (float(mp.mpf(x._mpi_[k])) for x, k in
                          ((lo1, 0), (hi1, 1), (lo2, 0), (hi2, 1)))
    report('Sec. 6.1', 'A(t)/t < 1.71 on [1.37, 1.38], > 2.96 on '
           '[2.49, 2.51] (certified)', '[1.691, 1.704]; [2.966, 2.972]',
           '[%.4f, %.4f]; [%.4f, %.4f]' % (lo1, hi1, lo2, hi2))
    t = np.linspace(1, 4, 30001)
    vals = {}
    for sign in (1, -1):
        X, Y = fnum.H(sign*t, 0*t)
        r = fnum.G_minus(X.real, Y.real)/t
        vals[sign] = (r.min(), r.max())
    report('Sec. 6.1', 'range of A(t)/t over a period; negative axis',
           '[1.670, 2.973]; [3.683, 3.748]',
           '[%.4f, %.4f]; [%.4f, %.4f]' % (vals[1] + vals[-1]))


def taylor():
    c = fnum.taylor_coefficients(4, exact=True)
    terms = {(1, 0): 1, (0, 2): '-4/3', (2, 1): '64/45',
             (4, 0): '-256/567', (1, 3): '-2048/2835'}
    got = {k: str(c[sum(k)].get(k, 0)) for k in terms}
    others = [k for d in c for k in c[d] if k not in terms]
    report('(8)', 'Taylor coefficients of x, y^2, x^2y, x^4, xy^3',
           ', '.join(str(v) for v in terms.values()),
           ', '.join(got.values()) + f'; no other terms: {not others}')


def real_ends():
    with mp.workdps(200):
        ends = []
        for lo, hi in [('0.8', '0.9'), ('-0.95', '-1.05')]:
            a, b = mp.mpf(lo), mp.mpf(hi)
            for _ in range(205):        # 2^-205 < 10^-61
                mid = (a + b)/2
                x, y = mid, mp.mpf(0)
                for _ in range(3000):
                    x, y = P(x, y)
                    if abs(x) > 1e6 or abs(x) + abs(y) < 1e-30:
                        break
                if abs(x) + abs(y) < 1e-30:
                    a = mid
                else:
                    b = mid
            x, y = a, mp.mpf(0)
            for _ in range(40):
                x, y = P(x, y)
            ends.append((a, x, y))
    report('Fig. 1, Sec. 2, Sec. 4.3', 'real segment of S',
           '(-0.9903, 0.8566)', '(%s, %s)' % (mp.nstr(ends[1][0], 10),
                                              mp.nstr(ends[0][0], 10)))


def window_max():
    xs = np.linspace(-3, 3, 900)
    X, Y = np.meshgrid(xs, xs)
    F = fnum.f(X, Y).real
    i = np.unravel_index(np.argmax(F), F.shape)
    report('Sec. 2', 'max of f on the window [-3,3]^2 of Fig. 1',
           'about 1.85', '%.3f at (%.2f, %.2f), 900^2 grid'
           % (F[i], X[i], Y[i]))


def supremum():
    """mu = sup f(R^2) = 2 sup{Y : (X, Y) in B, real} is the maximum
    of X over the real stable manifold of the saddle q, grown by
    backward iteration at 100 digits."""
    with mp.workdps(100):
        lam = (3 - mp.sqrt(13))/4
        q = (mp.mpf(3)/4, mp.mpf(3)/8)
        e = (mp.mpf(1), -(3 - 2*lam))
        s0, K, NB = mp.mpf('1e-40'), 1500, 70

        def ws_point(sgn, t, n):
            s = sgn*s0*abs(lam)**(-t)
            X, Y = q[0] + s*e[0], q[1] + s*e[1]
            for _ in range(n):
                X, Y = P_inv(X, Y)
            return X, Y

        best = (-mp.inf, None)
        for sgn in (1, -1):
            for i in range(K):
                s = sgn*s0*abs(lam)**(-mp.mpf(i)/K)
                X, Y = q[0] + s*e[0], q[1] + s*e[1]
                for n in range(1, NB + 1):
                    X, Y = P_inv(X, Y)
                    if X > best[0]:
                        best = (X, (sgn, mp.mpf(i)/K, n))
        sgn, t0, n = best[1]
        a, b = t0 - mp.mpf(2)/K, t0 + mp.mpf(2)/K
        xi = (mp.sqrt(5) - 1)/2
        c, d = b - xi*(b - a), a + xi*(b - a)
        for _ in range(220):
            if ws_point(sgn, c, n)[0] > ws_point(sgn, d, n)[0]:
                b, d = d, c
                c = b - xi*(b - a)
            else:
                a, c = c, d
                d = a + xi*(b - a)
        Xs, Ys = ws_point(sgn, (a + b)/2, n)
        x, y = Xs - mp.mpf('1e-20'), Ys
        for _ in range(4000):
            x, y = P(x, y)
            if abs(x) > 1e6 or abs(x) + abs(y) < mp.mpf('1e-60'):
                break
        inside = abs(x) + abs(y) < 1e-60
    report('Sec. 2', 'supremum of f on R^2',
           'about 3.444; 3.44401',
           '%s (max of X on the real stable manifold of q, at Y = %s; '
           '(X - 1e-20, Y) in the basin: %s)'
           % (mp.nstr(Xs, 12), mp.nstr(Ys, 8), inside))


def extent_of_S():
    for N in (1600, 3200):
        g = np.linspace(-1.3, 1.3, N)
        W = g[None, :] + 1j*g[::-1, None]
        inS = fnum.in_S(W)
        bd = inS & ~ndimage.binary_erosion(inS)
        r = abs(W[bd])
        report('Fig. 2', 'boundary of S, escape test on a %d^2 grid' % N,
               '0.857 <= |w| <= 1.063',
               '%.4f <= |w| <= %.4f (grid step %.1e); components: '
               'S %d, complement %d'
               % (r.min(), r.max(), g[1] - g[0], ndimage.label(inS)[1],
                  ndimage.label(~inS)[1]))
    th = np.linspace(0, 2*np.pi, 360, endpoint=False)
    rr = np.linspace(1e-4, 1.1, 4000)
    inS = fnum.in_S(rr[None, :]*np.exp(1j*th[:, None]), maxit=1500)
    first_out = rr[np.argmin(inS, axis=1)]
    reenter = sum(row[np.argmin(row):].any() for row in inS)
    report('Fig. 2',
           'largest disc about 0 in S, smallest containing S',
           'radius about 0.857; about 1.063',
           '%.4f; %.4f (first exit along 360 rays, step %.1e)'
           % (first_out.min(), first_out.max(), rr[1] - rr[0]))
    report('Sec. 4.4', 'rays from 0 leave S only once',
           'yes', 'rays that leave S and re-enter: %d of %d'
           % (reenter, len(th)))


def edges_and_periods(recompute):
    D = common.edge_data(recompute)
    up, lo = D[D[:, 0] == 1], D[D[:, 0] == -1]
    L = up[:, 1]
    xi = {'upper': up[:, 2], 'lower': lo[:, 2]}
    period = lambda k: (L >= k - 1e-9) & (L < k + 1 - 1e-9)
    rows = []
    for k in range(3, 7):
        a, b = period(k), period(k + 1)
        rows.append((abs(xi['upper'][b] - xi['upper'][a]).max(),
                     abs(xi['lower'][b] - xi['lower'][a]).max(),
                     abs(xi['upper'][a] - xi['lower'][a]).max()))
    rows = np.array(rows)
    amp = max(abs(xi[s][period(7)]).max() for s in xi)
    report('Fig. 1, Sec. 4.2', 'amplitude of y/x - 1 on the '
           'real edges', 'about 1.3e-4', '%.5e (on 7 <= log_4 x < 8)' % amp)
    return D


def accuracy_and_identities():
    rng = np.random.default_rng(1)
    z = rng.uniform(-7, 7, (60, 2))
    z = z[np.hypot(z[:, 0], z[:, 1]) <= 10]
    err = 0.0
    for a, b in z:
        exact = complex(fnum.f_mp(a, b, n=200, dps=200))
        approx = complex(fnum.f(np.array([a]), np.array([b]))[0])
        err = max(err, abs(approx - exact)/max(1, abs(exact)))
    report('Sec. 4.2', 'double-precision H, |z| <= 10', 'about twelve '
           'digits', 'max relative error %.1e at %d points' % (err, len(z)))

    with mp.workdps(200):
        f = fnum.f_mp
        x, y = mp.mpf('2'), mp.mpf('-1.4')
        lhs = f(x, y)
        eq = f(4*x, 4*y)/4 + f(2*y, 2*x)**2
        s = x - sum(mp.mpf(4)**j*f(2*y/mp.mpf(4)**j, 2*x/mp.mpf(4)**j)**2
                    for j in range(1, 60))
        g = lambda a, b: f(2*b, 2*a)/2
        d = lambda F, i: (lambda a, b: mp.diff(lambda t: F(t, b), a)
                          if i == 0 else mp.diff(lambda t: F(a, t), b))
        J = d(f, 0)(x, y)*d(g, 1)(x, y) - d(f, 1)(x, y)*d(g, 0)(x, y)
    report('Sec. 4.2', 'spot checks at (2, -1.4): f, (2), (20), det DH',
           'equal; equal; 1', 'f = %s; (2): %s; (20): %s; det DH = %s'
           % (mp.nstr(lhs, 15), mp.nstr(eq, 15), mp.nstr(s, 15),
              mp.nstr(J, 15)))


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Section 6.1 (Figure 4) and Section 6.4: Nevanlinna characteristics.
N_CHAR = 2000            # angles in [0, pi/3] (midpoint rule)


def _char(fun, r):
    """T(r) = mean of log+|fun| over |t| = r, for an entire fun with
    |fun(omega t)| = |fun(t)| and real coefficients (so [0, pi/3]
    suffices)."""
    phi = (np.arange(N_CHAR) + 0.5)*(np.pi/3)/N_CHAR
    v = np.abs(fun(r*np.exp(1j*phi)))
    return float(np.mean(np.log(np.maximum(v, 1.0))))


def _f0(t):
    return fnum.H(t, 0*t)[0]


E32_AT_1 = sum(1/factorial(3*n + 1) for n in range(20))   # E_{3,2}(1)


def _test_fE(t):
    """The test function f_E(t) = t E_{3,2}(1 - 8t^3)/E_{3,2}(1) of
    Sec. 6.5, where w E_{3,2}(w^3) =
    (e^w + om^2 e^{om w} + om e^{om^2 w})/3 for any cube root w of
    1 - 8t^3 (|t| >= 1 here)."""
    om = fnum.OMEGA
    w = (1 - 8*t**3 + 0j)**(1/3)
    return (t*sum(om**(-k)*np.exp(om**k*w) for k in range(3))
            / (3*w*E32_AT_1))


def _mean_A(r):
    """Mean of A(t) = G^-(H(t, 0)) over |t| = r, with
    G^- = 2^{-n}(log|Y_n| + log 8) once P^{-n} has escaped."""
    phi = (np.arange(N_CHAR) + 0.5)*(np.pi/3)/N_CHAR
    t = r*np.exp(1j*phi)
    X, Y = fnum.H(t, 0*t)
    G = np.zeros(t.shape)
    done = np.zeros(t.shape, bool)
    for n in range(1, 80):
        X, Y = 2*Y, 2*X - 8*Y*Y
        esc = (~done) & (np.abs(Y) > 1e100)
        G[esc] = (np.log(np.abs(Y[esc])) + np.log(8))/2.0**n
        done |= esc
        X[done], Y[done] = 0, 0
    return float(np.mean(G))


def _extremum(fun, lo, hi, sign):
    """Max (sign=+1) or min (sign=-1) of fun on [lo, hi]: a grid of
    201 points, then a bounded scalar search around the best point."""
    from scipy.optimize import minimize_scalar
    xs = np.linspace(lo, hi, 201)
    v = np.array([sign*fun(x) for x in xs])
    i = int(np.argmax(v))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    res = minimize_scalar(lambda x: -sign*fun(x), bounds=(a, b),
                          method='bounded', options={'xatol': 1e-7})
    return sign*max(v[i], -res.fun), res.x


def characteristic():
    """Sec. 6.1: T(4r) - 4T(r) - 3 log 4 and T(r) + log 4 - (mean of A
    on |t| = r)/2 are small for r >= 6 (largest near the dips of the
    profile); the nearly periodic profile of (T(r) + log 4)/r."""
    T = lambda r: _char(_f0, r)
    rs = np.exp(np.linspace(np.log(6), np.log(32), 60))
    dev = max(abs(T(4*r) - 4*T(r) - 3*np.log(4)) for r in rs)
    med = np.median([abs(T(4*r) - 4*T(r) - 3*np.log(4)) for r in rs])
    report('Sec. 6.1', '|T(4r) - 4T(r) - 3 log 4| for r >= 6',
           'less than 1e-4, much smaller away from the dips',
           'max %.1e (near the dip at log_4 r = 2.216), median %.1e, '
           'on 6 <= r <= 32' % (dev, med))
    prof = lambda s: (T(4.0**s) + np.log(4))/4.0**s
    pmin, smin = _extremum(prof, 1.5, 2.5, -1)
    pmax, smax = _extremum(prof, 1.5, 2.5, +1)
    report('Sec. 6.1', '(T(r) + log 4)/r over a period of log_4 r',
           'about 1.355 to 1.685',
           '%.4f (log_4 r = %.4f) to %.4f (log_4 r = %.4f)'
           % (pmin, smin, pmax, smax))
    d = [abs(T(r) + np.log(4) - _mean_A(r)/2)
         for r in (4.0**1.4, 4.0**1.8, 4.0**smin, 4.0**2.6)]
    report('Sec. 6.1', '|T(r) + log 4 - (mean of A over |t| = r)/2|',
           'less than 1e-4 for r >= 6',
           '%.1e, %.1e, %.1e (at the dip), %.1e at log_4 r = 1.4, 1.8, '
           '%.3f, 2.6' % (d[0], d[1], d[2], d[3], smin))
    return pmin, pmax


def test_functions(pmin, pmax):
    """Not in paper (cf. Sec. 6.5): the characteristic of the test
    function f_E(t) = t E_{3,2}(1 - 8t^3)/E_{3,2}(1), and its ratio to
    that of
    t -> f(t, 0)."""
    L = 3*np.sqrt(3)/np.pi
    TF = [_char(_test_fE, 4.0**s)/4.0**s for s in (2, 3, 3.5)]
    report('Not in paper (cf. Sec. 6.5)', 'T_fE(r)/r -> 3 sqrt 3/pi',
           '%.4f' % L,
           'T_fE(r)/r = %.4f, %.4f, %.4f at log_4 r = 2, 3, 3.5' % tuple(TF))
    report('Not in paper (cf. Sec. 6.5)',
           'apparent eventual range of T_fE/T_f',
           'between 0.98 and 1.22',
           '%.4f to %.4f (= 3 sqrt 3/pi over the profile extremes)'
           % (L/pmax, L/pmin))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--recompute', action='store_true',
                   help='recompute the caches in cache/')
    args = p.parse_args()
    t0 = time.time()
    taylor()
    real_ends()
    window_max()
    supremum()
    extent_of_S()
    edges_and_periods(args.recompute)
    ratio_on_V()
    growth_on_axis()
    accuracy_and_identities()
    test_functions(*characteristic())
    print('done in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()

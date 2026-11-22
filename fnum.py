"""Numerical tools for the paper "A Fatou-Bieberbach Definition of the
Reals That Does Not Come from Exponential Quotients", in the paper's
notation throughout.

    M(x, y) = (y/2, x/2)  (7),
    M^{-1}(x, y) = (2y, 2x),            M^2 = 1/4,
    T(x, y) = (x, y + 2x^2)  (a shear), P = M o T:
    P(x, y) = (x^2 + y/2, x/2),         P^{-1}(x, y) = (2y, 2x - 8y^2).

B (bold in the paper) is the basin of 0 under P and under Q = P o P.
K = lim 4^n Q^n = lim M^{-n} P^n on B (equations (14), (15) of the paper),
H = (f, g) = K^{-1}, g(z) = f(M^{-1} z)/2, and f is the entire function
with

    f(4x, 4y) = 4f(x, y) - 4f(2y, 2x)^2,  f(0,0) = 0,  Df(0,0) = (1, 0).

V_g = {z : f(4z) = 4f(z)} = {g = 0} = H^{-1}(C x {0}) is mapped by f onto
S = {f(z) : f(4z) = 4f(z)} = {w : (w, 0) in B},
with inverse w -> K(w, 0).  The figures use (u, v) = M^{-1} K(w, 0),
the zero of f over w, so that f(u, v) = 0 and f(v/2, u/2) = w.
Sets are bold in the paper (B, S, V_g, V_f, U) and plain here.

Main entry points
-----------------
P, P_inv, M, M_inv, T, T_inv, T_k_inv : the maps above
taylor_coefficients(N, exact)  : Taylor coefficients of f, recursion (11)
H(x, y), f(x, y), g(x, y)      : double precision, on arrays
H_mp(x, y), f_mp(x, y)         : multiprecision, shear product (18)-(19)
K(x, y)                        : forward limit (15), on arrays
K_w(w), K_w_mp(w)              : K(w, 0), scalar
uv_from_w(w)                   : (u, v) = M^{-1} K(w, 0) = K(0, 2w)
K_w_grid(W)                    : vectorized escape test for S and the
                                 sizes of u and v on a grid
in_S(W)                        : membership in S, on arrays
"""
from fractions import Fraction
import numpy as np
import mpmath as mp

OMEGA = np.exp(2j*np.pi/3)


# ------------------------------------------------------------------
# The maps.  They work on numbers, numpy arrays and mpmath numbers.
# ------------------------------------------------------------------
def M(x, y):
    """M(x, y) = (y/2, x/2), half the coordinate swap."""
    return y/2, x/2


def M_inv(x, y):
    """M^{-1}(x, y) = (2y, 2x)."""
    return 2*y, 2*x


def P(x, y):
    """The Henon map P(x, y) = (x^2 + y/2, x/2) = M(T(x, y))."""
    return x*x + y/2, x/2


def P_inv(x, y):
    """P^{-1}(X, Y) = (2Y, 2X - 8Y^2) = T^{-1}(M^{-1}(X, Y))."""
    return 2*y, 2*x - 8*y*y


def T(x, y):
    """The shear T(x, y) = (x, y + 2x^2), with P = M o T."""
    return x, y + 2*x*x


def T_inv(x, y):
    """T^{-1}(x, y) = (x, y - 2x^2)."""
    return x, y - 2*x*x


def T_k_inv(k, x, y):
    """Inverse of T_k = M^{-k} T M^k:

    T_{2j}(x, y) = (x, y + 2x^2/4^j),  T_{2j+1}(x, y) = (x + y^2/4^j, y).
    """
    j = k // 2
    q = mp.mpf(4)**j if isinstance(x, (mp.mpf, mp.mpc)) else 4.0**j
    if k % 2 == 0:
        return x, y - 2*x*x/q
    return x - y*y/q, y


def sigma(x, y):
    """sigma(x, y) = (omega x, omega^2 y); H o sigma = sigma o H."""
    return OMEGA*x, OMEGA**2*y


# ------------------------------------------------------------------
# Taylor coefficients, by the recursion (11):
#   c[a,b] (1 - 4^(d-1)) = [ sum_{i+j=d} h_i h_j ]_{a,b},  d = a + b,
# with h(x, y) = f(2y, 2x), so h_i(x, y) = 2^i f_i(y, x).
# ------------------------------------------------------------------
def taylor_coefficients(N, exact=False):
    """Homogeneous parts of f up to degree N, as {d: {(a, b): c}},
    c the coefficient of x^a y^b; exact rationals or floats."""
    one = Fraction(1) if exact else 1.0
    two, four = 2*one, 4*one
    f = {1: {(1, 0): one}}
    for d in range(2, N + 1):
        h = {i: {(b, a): c*two**i for (a, b), c in f[i].items()}
             for i in range(1, d)}
        R = {}
        for i in range(1, d):
            for (a, b), c in h[i].items():
                for (e, g_), c2 in h[d - i].items():
                    k = (a + e, b + g_)
                    R[k] = R.get(k, 0) + c*c2
        f[d] = {k: v/(1 - four**(d - 1)) for k, v in R.items() if v != 0}
    return f


_N = 40          # degree of the Taylor polynomial used by H
_R0 = 0.3        # H sums the series on the polydisc |x| <= R0, |y| <= 2 R0


def _coefficient_arrays(N):
    """Coefficient arrays of the degree-N Taylor polynomials of f and of
    g(x, y) = f(2y, 2x)/2."""
    Cf = np.zeros((N + 1, N + 1))
    Cg = np.zeros((N + 1, N + 1))
    for d, part in taylor_coefficients(N).items():
        for (a, b), c in part.items():
            Cf[a, b] += c
            Cg[b, a] += c*2.0**d/2
    return Cf, Cg


_CF, _CG = _coefficient_arrays(_N)


def _poly2(C, x, y):
    xs = np.stack([x**k for k in range(_N + 1)])
    ys = np.stack([y**k for k in range(_N + 1)])
    return np.einsum('ab,a...,b...->...', C, xs, ys)


def taylor_f(x, y):
    """The degree-40 Taylor polynomial of f (useful near 0 only)."""
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)
    return _poly2(_CF, x, y)


# ------------------------------------------------------------------
# H = (f, g) in double precision: Taylor series with backward
# iteration.  Since P o H = H o M, H = P^{-n} o H o M^n.
# ------------------------------------------------------------------
def H(x, y):
    """H(x, y) = (f, g) on arrays, double precision.

    Apply M until |x| <= 0.3 and |y| <= 0.6, sum the degree-40 Taylor
    series of f and g there, then apply P^{-1} as many times.  About 12
    digits for |(x, y)| up to about 10.
    """
    x = np.array(x, dtype=complex, ndmin=1)
    y = np.array(y, dtype=complex, ndmin=1)
    n = np.zeros(x.shape, int)
    for _ in range(400):
        big = np.maximum(abs(x), abs(y)/2) > _R0
        if not big.any():
            break
        x[big], y[big] = M(x[big], y[big])
        n[big] += 1
    X, Y = _poly2(_CF, x, y), _poly2(_CG, x, y)
    for k in range(n.max() if n.size else 0):
        m = n > k
        X[m], Y[m] = P_inv(X[m], Y[m])
    return X, Y


def f(x, y):
    """f on arrays, double precision (complex values)."""
    return H(x, y)[0]


def g(x, y):
    """g(z) = f(M^{-1} z)/2 on arrays, double precision."""
    return H(x, y)[1]


# ------------------------------------------------------------------
# H in multiprecision: the shear product (18)-(19),
#   P^{-n} o M^n = T_0^{-1} o T_1^{-1} o ... o T_{n-1}^{-1} -> H,
# with error O(2^-n).  The inverse shears amplify rounding, so many
# digits are needed.
# ------------------------------------------------------------------
def H_mp(x, y, n=200, dps=200):
    """H(x, y) = lim T_0^{-1} o ... o T_{n-1}^{-1} (x, y).

    Works at max(dps, current mpmath precision), so that mp.diff, which
    raises the working precision, sees its small steps.
    """
    with mp.workdps(max(dps, mp.mp.dps)):
        a, b = mp.mpmathify(x), mp.mpmathify(y)
        for k in reversed(range(n)):
            a, b = T_k_inv(k, a, b)
        return a, b


def f_mp(x, y, n=200, dps=200):
    """f(x, y) in multiprecision, via H_mp."""
    return H_mp(x, y, n, dps)[0]


def backward_limit_mp(x, y, n=200, dps=200):
    """P^{-n}(M^n (x, y)), computed literally (for cross-checks)."""
    with mp.workdps(max(dps, mp.mp.dps)):
        a, b = mp.mpmathify(x), mp.mpmathify(y)
        for _ in range(n):
            a, b = M(a, b)
        for _ in range(n):
            a, b = P_inv(a, b)
        return a, b


# ------------------------------------------------------------------
# The forward route.  For p in B, K(p) = lim M^{-n} P^n(p) (15).
# Iterating P toward the attracting fixed point is stable and M^{-1}
# is exact.  The orbit is stopped when |x| + |y|/2 < tol.
# ------------------------------------------------------------------
def K(x, y, tol=1e-15, maxit=3000, esc=1e6):
    """K(x, y) on arrays; nan where the orbit escapes (p not in B)."""
    x = np.array(x, dtype=complex, ndmin=1)
    y = np.array(y, dtype=complex, ndmin=1)
    n = np.zeros(x.shape, int)
    live = np.ones(x.shape, bool)
    with np.errstate(all='ignore'):
        for _ in range(maxit):
            live &= abs(x) + abs(y)/2 >= tol
            out = live & ~(abs(x) < esc)
            x[out] = y[out] = np.nan
            live &= ~out
            if not live.any():
                break
            x[live], y[live] = P(x[live], y[live])
            n[live] += 1
        for k in range(n.max() if n.size else 0):
            m = n > k
            x[m], y[m] = M_inv(x[m], y[m])
    return x, y


def _uv_loop(a, b, tol, maxit, esc):
    """Iterate P from (a, b) = (w, 0) and apply M^{-1} once more than
    the number of steps: returns M^{-1} K(w, 0), or None on escape."""
    k = 1
    while abs(a) + abs(b)/2 > tol and k < maxit:
        a, b = P(a, b)
        k += 1
        if esc is not None and not abs(a) < esc:
            return None
    for _ in range(k):
        a, b = M_inv(a, b)
    return a, b


def uv_from_w(w, tol=1e-14, maxit=3000):
    """(u, v) = M^{-1} K(w, 0) = K(0, 2w): the zero of f over w in S,
    with f(u, v) = 0 and f(v/2, u/2) = w.  Double precision for numbers,
    multiprecision for mpmath input (use inside mp.workdps); None if
    the orbit of (w, 0) escapes."""
    if isinstance(w, (mp.mpf, mp.mpc)):
        return K_w_mp(w, uv=True)
    return _uv_loop(complex(w), 0j, tol, maxit, 1e6)


def K_w(w, tol=1e-14, maxit=3000):
    """K(w, 0) in double precision: the point of Z over w, or None."""
    r = uv_from_w(w, tol, maxit)
    return None if r is None else M(*r)


def K_w_mp(w, tol_exp=45, uv=False):
    """K(w, 0) in multiprecision (or M^{-1} K(w, 0) if uv); use inside
    mp.workdps."""
    r = _uv_loop(w, mp.mpf(0)*w, mp.mpf(10)**(-tol_exp), 20000, None)
    return r if uv else M(*r)


def K_w_grid(W, tol=1e-11, maxit=400, esc=1e6):
    """Vectorized escape test of (w, 0) for an array W.

    Returns (inS, U, V, log2|u|, log2|v|), where (u, v) = M^{-1} K(w, 0)
    and U, V equal u, v up to the positive factor 4^(k//2) for the number
    k of applications of M^{-1} (so they have the arguments of u, v).
    Outside S the logarithms are nan.  Points that neither converge nor
    escape within maxit steps are undetermined and counted as outside.
    """
    x = np.array(W, dtype=complex)
    y = np.zeros_like(x)
    n = np.zeros(x.shape, int)
    st = np.zeros(x.shape, int)              # 0 live, 1 basin, 2 escape
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    with np.errstate(all='ignore'):
        for k in range(1, maxit):
            if k > 1:
                x, y = P(x, y)
            live = st == 0
            conv = live & (abs(x) + abs(y)/2 < tol)
            out = live & ~(abs(x) < esc)
            st[conv], n[conv] = 1, k
            px[conv], py[conv] = x[conv], y[conv]
            st[out] = 2
            x = np.where(st == 0, x, 0)
            y = np.where(st == 0, y, 0)
            if not (st == 0).any():
                break
    inS = st == 1
    odd = n % 2 == 1                         # M^{-2} = 4 I
    U = np.where(odd, 2*py, px)
    V = np.where(odd, 2*px, py)
    with np.errstate(all='ignore'):
        Lu = np.where(inS, np.log2(abs(U) + 1e-300) + 2*(n//2), np.nan)
        Lv = np.where(inS, np.log2(abs(V) + 1e-300) + 2*(n//2), np.nan)
    return inS, U, V, Lu, Lv


def in_S(W, **kw):
    """Membership in S = {w : (w, 0) in B}, on arrays.  Points whose
    orbit neither converges nor escapes within the iteration limit are
    undetermined; they are classified as outside S, as for plotting."""
    return K_w_grid(W, **kw)[0]


# ------------------------------------------------------------------
# Curves in S on which u (or v) is real, traced from the zeros of u
# by Newton continuation.  comp = 0 for u, 1 for v.
# ------------------------------------------------------------------
def comp_of(w, comp=0):
    """u(w) (comp 0) or v(w) (comp 1) for (u, v) = M^{-1} K(w, 0)."""
    r = uv_from_w(w)
    return None if r is None else r[comp]


def d_comp(w, comp=0, h=1e-8):
    a, b = comp_of(w + h, comp), comp_of(w - h, comp)
    return None if a is None or b is None else (a - b)/(2*h)


def newton(w, target, comp=0, it=50):
    """Solve u(w) = target (or v(w) = target) by Newton's method."""
    for _ in range(it):
        f0, d = comp_of(w, comp), d_comp(w, comp)
        if f0 is None or d is None or d == 0:
            return None
        step = (f0 - target)/d
        w -= step
        if abs(step) < 1e-15*max(1, abs(w)):
            break
    f0 = comp_of(w, comp)
    ok = f0 is not None and abs(f0 - target) < 1e-9*max(1, abs(target))
    return w if ok else None


def zeros_on_grid(xs, inS, L2, comp=0):
    """Zeros of u (or v) in S: local minima of log2|u| on the grid
    xs + i xs (rows from the top), refined by secant steps."""
    from scipy.ndimage import minimum_filter
    m = np.where(inS, L2, np.inf)
    i, j = np.where((m == minimum_filter(m, size=15)) & np.isfinite(m))
    out = []
    for w in xs[j] + 1j*xs[::-1][i]:
        for _ in range(40):
            f0, f1 = comp_of(w, comp), comp_of(w + 1e-7, comp)
            if f0 is None or f1 is None or f1 == f0:
                break
            step = f0*1e-7/(f1 - f0)
            w -= step
            if abs(step) < 1e-13:
                break
        f0 = comp_of(w, comp)
        if f0 is not None and abs(f0) < 1e-9 and \
                all(abs(w - z) > 1e-6 for z in out):
            out.append(w)
    return np.array(out)


def trace(z0, w1, sign, factor=1.15):
    """Follow the curve {u real} from w1 with |u| increasing by the
    given factor per step until |u| > 1e15; returns [z0, w1, ...]."""
    w, val, path = w1, abs(comp_of(w1)), [z0, w1]
    while val < 1e15:
        val *= factor
        w2 = newton(w, sign*val)
        if w2 is None or abs(w2 - w) > 0.05:
            break
        w = w2
        path.append(w)
    return np.array(path)


def branches(z0, double=False, eps=1e-3, factor=1.15):
    """The curves {u real} starting at a zero z0 of u, as a list of
    (sign of u, path).  At the double zero w = 0, u ~ (16/3) w^2."""
    if double:
        starts = [(eps, 1), (-eps, 1), (1j*eps, -1), (-1j*eps, -1)]
    else:
        d = 1/d_comp(z0)
        d /= abs(d)                           # u increases along d
        starts = [(z0 + eps*d, 1), (z0 - eps*d, -1)]
    out = []
    for w1, s in starts:
        w1 = newton(w1, s*abs(comp_of(w1)))
        if w1 is not None:
            out.append((s, trace(z0, w1, s, factor)))
    return out


# ------------------------------------------------------------------
# The real basin and the flow lines of Q = P^2 (companion paper).
# Since M^2 = 1/4, Q^s = H o 4^{-s} o K is a flow on B with
# Q^1 = P^2; its flow lines are the images H(R_phi) of the rays
# R_phi = {r (cos phi, sin phi) : r > 0}.
# ------------------------------------------------------------------
def in_basin(x, y, maxit=800, tol=1e-8, esc=1e8):
    """Membership in B, by escape testing, on arrays.  Points whose
    orbit neither converges nor escapes within maxit steps are
    undetermined; they are classified as outside B, as for plotting."""
    x = np.array(x, dtype=float if np.isrealobj(x) else complex)
    y = np.array(y, dtype=x.dtype)
    st = np.zeros(np.broadcast(x, y).shape, int)  # 0 live, 1 in, 2 out
    x, y = np.broadcast_arrays(x, y)
    x, y = x.copy(), y.copy()
    with np.errstate(all='ignore'):
        for _ in range(maxit):
            x, y = P(x, y)
            st[(st == 0) & (abs(x) + abs(y) < tol)] = 1
            st[(st == 0) & ~(abs(x) < esc)] = 2
            x = np.where(st == 0, x, 0)
            y = np.where(st == 0, y, 0)
            if not (st == 0).any():
                break
    return st == 1


def flow_line(phi, n=3000, kmin=-6, kmax=3, dps=120, big=1e6):
    """The flow line H(R_phi), sampled at r = 4^(k + j/n),
    kmin <= k <= kmax, 0 <= j < n.

    Only the seed period 1 <= r < 4 is evaluated by the shear product
    (H_mp); since H(4z) = P^{-2}(H(z)) and H(z/4) = P^2(H(z)), the
    other periods follow by applying P^{-2} outward and P^2 inward, two
    polynomial steps per point.  Returns (log4 r, X, Y) as float
    arrays, with nan where max(|X|, |Y|) > big.
    """
    with mp.workdps(dps):
        th = mp.mpf(phi)
        c, s = mp.cos(th), mp.sin(th)
        seed = []
        for j in range(n):
            r = mp.mpf(4)**(mp.mpf(j)/n)
            seed.append(H_mp(r*c, r*s, n=260, dps=dps))
        periods = {0: seed}
        for k in range(1, kmax + 1):
            periods[k] = [P_inv(*P_inv(a, b)) for a, b in periods[k - 1]]
        for k in range(-1, kmin - 1, -1):
            periods[k] = [P(*P(a, b)) for a, b in periods[k + 1]]
        L, X, Y = [], [], []
        for k in range(kmin, kmax + 1):
            for j, (a, b) in enumerate(periods[k]):
                L.append(k + j/n)
                ok = abs(a) < big and abs(b) < big
                X.append(float(a) if ok else np.nan)
                Y.append(float(b) if ok else np.nan)
    return np.array(L), np.array(X), np.array(Y)


def unstable_branch(n=300, steps=60, side=-1):
    """A branch of the real unstable manifold of the saddle
    q = (3/4, 3/8): a fundamental domain along the unstable
    eigenvector, iterated by P.  side=-1 gives the branch U that runs
    into 0, side=+1 the branch U' that leaves through x > 3/4 (take
    steps small enough to avoid overflow).  Returns an (m, 2) array."""
    lu = (3 + np.sqrt(13))/4
    e = np.array([1, 1/(2*lu)])
    e /= np.hypot(*e)
    s = side*1e-7*np.exp(np.linspace(0, np.log(lu), n))
    x, y = .75 + s*e[0], .375 + s*e[1]
    out = []
    for _ in range(steps):
        out.append(np.column_stack([x, y]))
        x, y = P(x, y)
    return np.vstack(out)


def G_minus(X, Y, nmax=80, big=1e30):
    """Backward Green function of P (equal to that of Q = P o P), by
    P^{-1}(X, Y) = (2Y, 2X - 8Y^2) and, once |Y_n| > big,
    G ~ 2^{-n} (log|Y_n| + log 8).  Zero where no escape occurs within
    nmax steps (bounded backward orbit).  Double precision; numerical
    illustration only."""
    X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
    X, Y = X.copy(), Y.copy()
    G = np.zeros(X.shape)
    done = np.zeros(X.shape, bool)
    for n in range(nmax):
        act = ~done
        X[act], Y[act] = 2*Y[act], 2*X[act] - 8*Y[act]**2
        esc = act & (np.abs(Y) > big)
        G[esc] = (np.log(np.abs(Y[esc])) + np.log(8))/2.0**(n + 1)
        done |= esc
        if done.all():
            break
    return G

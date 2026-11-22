"""Identities of the paper: the Taylor coefficients and polynomial
identities in exact rational arithmetic, the rest numerically
(about 20 s).

    python -m pytest
"""
from fractions import Fraction
import numpy as np
import mpmath as mp
import pytest
import fnum

OMEGA = fnum.OMEGA
rng = np.random.default_rng(0)
# sample points in the bidisc of radius 2 (double precision H is good
# to about 1e-13 there)
Z = rng.uniform(-2, 2, (40, 2)) + 1j*rng.uniform(-2, 2, (40, 2))
X, Y = Z[:, 0], Z[:, 1]


def close(a, b, tol):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(abs(a - b)/np.maximum(1, abs(b))) < tol


# ---------------------------------------------------------- the maps
def test_P_is_M_after_T():
    assert close(fnum.P(X, Y), fnum.M(*fnum.T(X, Y)), 1e-15)


def test_P_inverse():
    assert close(fnum.P_inv(*fnum.P(X, Y)), (X, Y), 1e-13)
    assert close(fnum.P_inv(X, Y), fnum.T_inv(*fnum.M_inv(X, Y)), 1e-15)


def test_shears_are_conjugates_of_T():
    """T_k = M^{-k} T M^k has the closed forms of (18)."""
    for k in range(6):
        a, b = X, Y
        for _ in range(k):
            a, b = fnum.M(a, b)
        a, b = fnum.T_inv(a, b)
        for _ in range(k):
            a, b = fnum.M_inv(a, b)
        assert close((a, b), fnum.T_k_inv(k, X, Y), 1e-13)


def test_Q_inverse_is_two_shears():
    """Q^{-1}(z/4) = T_0^{-1} T_1^{-1} z, and conjugating T_k by 4^j
    gives T_{k+2j}."""
    a, b = fnum.P_inv(*fnum.P_inv(X/4, Y/4))
    assert close((a, b), fnum.T_k_inv(0, *fnum.T_k_inv(1, X, Y)), 1e-13)
    for k in range(4):
        for j in range(3):
            a, b = fnum.T_k_inv(k, X/4**j, Y/4**j)
            assert close((4**j*a, 4**j*b), fnum.T_k_inv(k + 2*j, X, Y),
                         1e-13)


def test_shear_product_is_backward_limit():
    """P^{-n} o M^n = T_0^{-1} o ... o T_{n-1}^{-1}."""
    with mp.workdps(60):
        for x, y in Z[:3]:
            a = fnum.H_mp(complex(x), complex(y), n=12, dps=60)
            b = fnum.backward_limit_mp(complex(x), complex(y), n=12,
                                       dps=60)
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) < 1e-40


# ---------------------------------------------------- f, H, K
def test_functional_equation_double():
    """(2): f(z) = f(4z)/4 + f(M^{-1} z)^2."""
    lhs = fnum.f(X/4, Y/4)
    rhs = fnum.f(X, Y)/4 + fnum.f(*fnum.M_inv(X/4, Y/4))**2
    assert close(lhs, rhs, 1e-12)


def test_functional_equation_mp():
    with mp.workdps(200):
        x, y = mp.mpf('2'), mp.mpf('-1.4')
        f = fnum.f_mp
        assert abs(f(x, y) - f(4*x, 4*y)/4 - f(2*y, 2*x)**2) < 1e-40


def test_g_from_f():
    """g(z) = f(M^{-1} z)/2."""
    assert close(fnum.g(X, Y), fnum.f(*fnum.M_inv(X, Y))/2, 1e-12)


def test_conjugacy_P_H_equals_H_M():
    """P o H = H o M."""
    assert close(fnum.P(*fnum.H(X, Y)), fnum.H(*fnum.M(X, Y)), 1e-12)


def test_K_inverts_H():
    """K o H = id."""
    small = np.max(abs(Z), axis=1) < 1.5
    x, y = X[small], Y[small]
    assert close(fnum.K(*fnum.H(x, y)), (x, y), 1e-10)


def test_K_conjugacy():
    """K o P = M o K on the basin."""
    x, y = fnum.H(X, Y)
    assert close(fnum.K(*fnum.P(x, y)), fnum.M(*fnum.K(x, y)), 1e-10)


def test_orbit_recurrence():
    """x_n = f(M^n z) satisfies x_{n+1} = x_n^2 + x_{n-1}/4, and
    H(M^n z) = (x_n, x_{n-1}/2)."""
    x = []
    for n in range(-1, 7):
        a, b = X, Y
        for _ in range(abs(n)):
            a, b = (fnum.M if n > 0 else fnum.M_inv)(a, b)
        x.append(fnum.H(a, b))
    for k in range(1, len(x) - 1):
        assert close(x[k + 1][0], x[k][0]**2 + x[k - 1][0]/4, 1e-9)
        assert close(x[k][1], x[k - 1][0]/2, 1e-9)


# ---------------------------------------------------- Taylor series
def test_taylor_start():
    c = fnum.taylor_coefficients(4, exact=True)
    assert c[1] == {(1, 0): 1}
    assert c[2] == {(0, 2): Fraction(-4, 3)}
    assert c[3] == {(2, 1): Fraction(64, 45)}
    assert c[4] == {(4, 0): Fraction(-256, 567),
                    (1, 3): Fraction(-2048, 2835)}


def test_taylor_satisfies_functional_equation():
    """The recursion (11) makes the truncated series satisfy (2) up to
    the truncation degree, exactly in rationals."""
    N = 9
    c = fnum.taylor_coefficients(N, exact=True)
    f = {k: v for d in c for k, v in c[d].items()}
    h = {(b, a): v*2**(a + b) for (a, b), v in f.items()}
    sq = {}
    for (a, b), p in h.items():
        for (e, g), q in h.items():
            if a + b + e + g <= N:
                k = (a + e, b + g)
                sq[k] = sq.get(k, 0) + p*q
    for (a, b), v in f.items():
        assert v == Fraction(4)**(a + b - 1)*v + sq.get((a, b), 0)


def test_taylor_against_series():
    """The Taylor polynomial against the shear-product limit (19) and
    against the series (20), near 0."""
    pts = [(0.05, -0.03), (0.02 + 0.04j, -0.05j), (-0.06, 0.06)]
    with mp.workdps(80):
        for x, y in pts:
            t = complex(fnum.taylor_f(x, y))
            lim = complex(fnum.f_mp(x, y, n=200, dps=200))
            ser = complex(x) - sum(
                4.0**j*complex(fnum.f_mp(2*y/4**j, 2*x/4**j, n=200,
                                         dps=200))**2
                for j in range(1, 40))
            assert abs(t - lim) < 1e-14
            assert abs(lim - ser) < 1e-14


# ---------------------------------------------------- symmetry
def test_monomials_threefold():
    c = fnum.taylor_coefficients(12, exact=True)
    assert all((a + 2*b) % 3 == 1 for d in c for (a, b) in c[d])


def test_H_commutes_with_sigma():
    """H o sigma = sigma o H, M o sigma = sigma^2 o M,
    P o sigma = sigma^2 o P."""
    s = fnum.sigma
    s2 = lambda x, y: s(*s(x, y))
    assert close(fnum.H(*s(X, Y)), s(*fnum.H(X, Y)), 1e-12)
    assert close(fnum.M(*s(X, Y)), s2(*fnum.M(X, Y)), 1e-15)
    assert close(fnum.P(*s(X, Y)), s2(*fnum.P(X, Y)), 1e-14)


def test_S_threefold():
    w = rng.uniform(-1.2, 1.2, 4000) + 1j*rng.uniform(-1.2, 1.2, 4000)
    a = fnum.in_S(w)
    assert (a == fnum.in_S(OMEGA*w)).mean() > 0.999
    assert (a == fnum.in_S(np.conj(w))).mean() > 0.999


# ---------------------------------------------------- S
def test_bounds_on_S():
    """D(0, 3/4) is in S and S is in D(0, 5/4), on a sample."""
    r = np.sqrt(rng.uniform(0, 1, 20000))
    w = r*np.exp(2j*np.pi*rng.uniform(0, 1, r.size))
    assert fnum.in_S(0.749*w).all()
    assert not fnum.in_S(w*1.251/abs(w)*(1 + r)).any()


def test_zero_over_w():
    """For w in S, (u, v) = M^{-1} K(w, 0) is a zero of f with
    f(v/2, u/2) = w, and K(w, 0) is the point of Z over w."""
    for w in (0.3, -0.5 + 0.2j, 0.1 - 0.6j, 0.7j):
        u, v = fnum.uv_from_w(w)
        a, b = fnum.K_w(w)
        assert abs(fnum.f(u, v)[0]) < 1e-11
        assert abs(fnum.f(v/2, u/2)[0] - w) < 1e-12
        assert abs(fnum.g(a, b)[0]) < 1e-12
        assert abs(fnum.f(a, b)[0] - w) < 1e-12
        # on Z, (2) reads f(4z) = 4 f(z)
        assert abs(fnum.f(4*a, 4*b)[0] - 4*w) < 1e-10


@pytest.mark.parametrize('w', [0.2, -0.4 + 0.3j])
def test_vertical_slice(w):
    """P(0, 2w) = (w, 0), so K(0, 2w) = M^{-1} K(w, 0)."""
    assert fnum.P(0, 2*w) == (w, 0)
    a, b = fnum.K(0, 2*w)
    u, v = fnum.uv_from_w(w)
    assert abs(a[0] - u) + abs(b[0] - v) < 1e-12


def test_growth_on_axis_not_regular():
    """Sec. 6.1: the certificate of verify_growth.py on two short pieces,
    including its checks that the iterates escape."""
    from mpmath import iv
    import verify_growth
    lo1, hi1 = verify_growth.piece(iv.mpf(['1.3749', '1.3751']))
    lo2, hi2 = verify_growth.piece(iv.mpf(['2.4999', '2.5001']))
    assert hi1 < iv.mpf('1.71').a and lo2 > iv.mpf('2.96').b



def test_double_precision_far_out():
    """At the edge of the range used: H in double precision against the
    multiprecision shear product at |t| = 100 on the line y = 0."""
    for th in (0.3, 1.9, 4.4):
        t = 100*np.exp(1j*th)
        X, Y = fnum.H(t, 0*t)
        Xm, Ym = fnum.H_mp(mp.mpc(t.real, t.imag), 0, n=200, dps=60)
        assert abs(X[0] - complex(Xm)) <= 1e-12*abs(complex(Xm))
        assert abs(Y[0] - complex(Ym)) <= 1e-12*abs(complex(Ym))

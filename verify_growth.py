"""Certified illustration for Section 6.1: the backward escape rate
A(t) = G^-(H(t, 0)) is far from homogeneous.  Theorem 6.3 (t -> f(t, 0)
does not have completely regular growth) has a proof without
computation; this script certifies the oscillation that suggested it,
and gives a second, computational proof via the first step of the
proof of Theorem 6.3.  Run (under a second; no caches, no other
modules of this repository):

    python verify_growth.py

It prints enclosures of A(t)/t, where A(t) = G^-(H(t, 0)) and G^- is the
backward escape rate of Q = P o P in the maximum norm, on the intervals
[1.37, 1.38] and [2.49, 2.51], and checks that the first lies below
1.71 and the second above 2.96, so that A(t)/t is not constant.

Arithmetic.  All quantities are intervals in mpmath's iv context
(mpmath 1.4.1 was used), at 50 decimal digits.  Every operation of
this context, including log, rounds its endpoints outward, so each
interval contains the exact value of the quantity it represents.  The
input intervals are rounded outward from the decimal strings, and the
final comparisons with 1.71 and 2.96 are made between interval
endpoints, never after conversion to float.

Steps, with Q^{-1}(X, Y) = (4Z, 4Y - 32 Z^2), Z = X - 4Y^2:

1. Starting enclosure.  For |w| < 3/112 in the maximum norm,
   |H(w) - w| <= (448/9)|w|^2.  Derivation (maximum norm throughout):
   Q(p) = p/4 + O(|p|^2) gives |4Q(p) - p| <= 4|p|^2 for |p| <= 1/2,
   so |Q(p)| <= (3/8)|p| for |p| <= 1/8.  Consecutive terms of
   K(p) = lim 4^n Q^n(p) differ by 4^n (4Q - 1)(Q^n p), so
   |K(p) - p| <= sum_n 4^n * 4|Q^n p|^2 <= (64/7)|p|^2 for |p| <= 1/8,
   and |K(p)| >= (3/7)|p| for |p| <= 1/16.  For |w| < 3/112, H maps
   the segment from 0 to w into |p| < 1/16: a first point s*w of it
   with |H(s*w)| = 1/16 would have |s*w| = |K(H(s*w))| >= 3/112.  So
   |H(w)| <= (7/3)|w| and
   |H(w) - w| = |K(H(w)) - H(w)| <= (64/7)(7/3)^2|w|^2
   = (448/9)|w|^2.  With
   w = (t, 0)/4^20, H(w) lies in the box (t/4^20 +- eps, +- eps),
   eps = (448/9)(t/4^20)^2.
2. Subdivision.  Each of the two t-intervals is cut into 20 closed
   pieces; the box of step 1 is formed for each piece with t an
   interval, so every t in the piece is covered.
3. Twenty steps of Q^{-1} give an enclosure of H(t, 0), since
   H(4z) = Q^{-1}(H(z)).  The script checks that it lies in the region
   |Y| >= max(|X|, 1).
4. Escape region.  There 3Y^2 <= |Z| <= 5Y^2, so
   284|Y|^4 <= |Y'| <= 804|Y|^4 and |Y'| >= max(|X'|, 1) for
   (X', Y') = Q^{-1}(X, Y); the region is invariant and
   G^-(X, Y) - log|Y| lies between (log 284)/3 and (log 804)/3.
5. Twelve more steps of Q^{-1} (the script checks that 0 is still
   outside Y), and G^-(Q^{-1} p) = 4 G^-(p), give
   A(t)/t >= (log min|Y| + (log 284)/3)/(4^12 t) and
   A(t)/t <= (log max|Y| + (log 804)/3)/(4^12 t).
6. The minimum and maximum over the 20 pieces give the enclosures,
   printed to 8 digits as [1.6912613, 1.7031209] and
   [2.9661488, 2.9714061]; the comparisons use the exact endpoints.
"""
from mpmath import iv, mpf, nstr

iv.dps = 50
N, EXTRA, PIECES = 20, 12, 20


def qinv(X, Y):
    Z = X - 4*Y**2
    return 4*Z, 4*Y - 32*Z**2


def absmin(I):
    return 0 if I.a <= 0 <= I.b else min(abs(I.a), abs(I.b))


def absmax(I):
    return max(abs(I.a), abs(I.b))


def piece(t):
    """Interval endpoints (lo, hi) with A(s)/s in [lo, hi] for s in t."""
    w = t/iv.mpf(4)**N
    assert absmax(w) < iv.mpf(3)/112
    eps = (iv.mpf(448)/9*absmax(w)**2).b
    X, Y = w + iv.mpf([-eps, eps]), iv.mpf([-eps, eps])
    for _ in range(N):
        X, Y = qinv(X, Y)
    assert absmin(Y) >= 1 and absmin(Y) >= absmax(X), 'not escaping'
    for _ in range(EXTRA):
        X, Y = qinv(X, Y)
    assert absmin(Y) >= 1, '0 in Y'
    s = iv.mpf(4)**EXTRA
    lo = (iv.log(absmin(Y)) + iv.log(284)/3)/s/t
    hi = (iv.log(absmax(Y)) + iv.log(804)/3)/s/t
    return lo.a, hi.b


def cover(a, b, m=PIECES):
    """Enclosure of A(t)/t for t in [a, b], a and b decimal strings."""
    A, B = iv.mpf(a).a, iv.mpf(b).b
    pts = [A + (B - A)*k/m for k in range(m)] + [B]
    assert all(pts[k] <= pts[k + 1] for k in range(m))
    out = [piece(iv.mpf([pts[k], pts[k + 1]])) for k in range(m)]
    return min(x for x, _ in out), max(y for _, y in out)


def certify():
    lo1, hi1 = cover('1.37', '1.38')
    lo2, hi2 = cover('2.49', '2.51')
    ok = hi1 < iv.mpf('1.71').a and lo2 > iv.mpf('2.96').b
    return (lo1, hi1), (lo2, hi2), ok


def main():
    (lo1, hi1), (lo2, hi2), ok = certify()
    for (a, b), (lo, hi) in ((('1.37', '1.38'), (lo1, hi1)),
                             (('2.49', '2.51'), (lo2, hi2))):
        print('A(t)/t for t in [%s, %s] lies in [%s, %s]'
              % (a, b, nstr(mpf(lo._mpi_[0]), 8),
                 nstr(mpf(hi._mpi_[1]), 8)))
    print('A(t)/t < 1.71 on the first and > 2.96 on the second:', ok)
    assert ok


if __name__ == '__main__':
    main()

from fractions import Fraction

def truncate(f, t):
    if t == 0:
        return 0
    mask = (1 << t) - 1
    f = f & mask
    if f >= (1 << (t-1)):
        f -= (1 << t)
    return f

def sign(x):
    return 1 if x >= 0 else -1

def divsteps2(n, t, delta, f, g):
    f, g = truncate(f, t), truncate(g, t)
    u, v, q, r = Fraction(1), Fraction(0), Fraction(0), Fraction(1)
    while n > 0:
        f = truncate(f, t)
        if delta > 0 and (g & 1):
            delta, f, g, u, v, q, r = -delta, g, -f, q, r, -u, -v
        g0 = g & 1
        delta = 1 + delta
        g = (g + g0 * f) // 2
        q = (q + Fraction(g0) * u) / 2
        r = (r + Fraction(g0) * v) / 2
        n, t = n - 1, t - 1
        g = truncate(g, t)
    return delta, f, g, ((u, v), (q, r))

def iterations(d):
    return (49*d + 80)//17 if d < 46 else (49*d + 57)//17

def recip2(f, g):
    assert f & 1
    d = max(f.bit_length(), g.bit_length())
    m = iterations(d)
    precomp = pow((f + 1)//2, m - 1, f)
    _, fm, _, P = divsteps2(m, m + 1, 1, f, g)
    (_, v_frac), _ = P
    V_int = int(v_frac * (1 << (m - 1))) * sign(fm)
    inv = (V_int * precomp) % f
    return inv
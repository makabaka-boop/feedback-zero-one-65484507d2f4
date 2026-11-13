import sys
sys.path.insert(0, '/workspace')
from fractions import Fraction
from itertools import product

# ---- independent reference: enumerate center states (free, +1, -1) ----
def Dmat(n):
    m = n - 2
    D = [[Fraction(0)] * n for _ in range(m)]
    for i in range(m):
        D[i][i] = Fraction(1); D[i][i+1] = Fraction(-2); D[i][i+2] = Fraction(1)
    return D

def gauss(A, b):
    m = len(A)
    M = [list(A[i]) + [b[i]] for i in range(m)]
    for col in range(m):
        p = next(r for r in range(col, m) if M[r][col] != 0)
        M[col], M[p] = M[p], M[col]
        for r in range(m):
            if r == col: continue
            f = M[r][col] / M[col][col]
            for k in range(col, m+1):
                M[r][k] -= f * M[col][k]
    return [M[i][m] / M[i][i] for i in range(m)]

def brute_fit(y, lam):
    n = len(y); m = n - 2
    D = Dmat(n)
    G = [[sum(D[k][i]*D[l][i] for i in range(n)) for l in range(m)] for k in range(m)]
    Dy = [sum(D[c][i]*y[i] for i in range(n)) for c in range(m)]
    feasible = []
    for state in product((0, 1, -1), repeat=m):  # 0=free, +/-1=bound
        free = [c for c in range(m) if state[c] == 0]
        bound = [c for c in range(m) if state[c] != 0]
        s = [Fraction(state[c]) for c in range(m)]
        ok = True
        if free:
            A = [[lam*G[c][d] for d in free] for c in free]
            b = [Dy[c] - sum(lam*G[c][a]*Fraction(state[a]) for a in bound) for c in free]
            q = gauss(A, b)
            for c, val in zip(free, q):
                if not (-1 <= val <= 1): ok = False
                s[c] = val
        if not ok: continue
        d = [Dy[c] - lam*sum(G[c][k]*s[k] for k in range(m)) for c in range(m)]
        for c in bound:
            if state[c] * d[c] <= 0:  # strict sign needed at active bound
                ok = False; break
        for c in free:
            if d[c] != 0: ok = False; break
        if ok:
            Dt = [[D[k][i] for k in range(m)] for i in range(n)]
            x = [y[i] - lam*sum(Dt[i][k]*s[k] for k in range(m)) for i in range(n)]
            obj = sum((y[i]-x[i])**2 for i in range(n))/2 + lam*sum(abs(v) for v in d)
            feasible.append((state, tuple(s), tuple(x), obj))
    x0 = feasible[0][2]
    knots = tuple(c+1 for c in range(m) if x0[c]-2*x0[c+1]+x0[c+2] != 0)
    return feasible, knots, x0

def show(label, y, lam):
    feas, knots, x = brute_fit(y, lam)
    print(label, 'knots=', knots, 'x=', x, '#feasible states=', len(feas))
    return feas, knots, x

# multi-knot search
import random
rng = random.Random(7)
found_multi = None
for _ in range(2000):
    n = rng.choice((5,6))
    y = [Fraction(rng.randint(-15,15)) for _ in range(n)]
    lam = Fraction(rng.randint(1,4))
    feas, knots, x = brute_fit(y, lam)
    if len(knots) >= 2:
        found_multi = (y, lam, knots, x)
        break
print('multi-knot example:', found_multi)

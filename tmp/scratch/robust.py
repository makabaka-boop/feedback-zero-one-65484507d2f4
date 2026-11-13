import sys, random
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tmp/scratch')
from fractions import Fraction
from explore import brute_fit
from seg_infer.trend import fit_trend, run_trend_inference

# 1) all-equal y -> no knots, no_change
r = run_trend_inference([Fraction(5)]*5, Fraction(2), Fraction(1), Fraction(0), Fraction(1,20))
print('all equal:', r.changed, r.knots)

# 2) huge lambda -> affine line, no knots
y = [Fraction(i*i) for i in range(6)]
r = run_trend_inference(y, Fraction(10**6), Fraction(1), Fraction(0), Fraction(1,20))
print('huge lambda knots:', r.knots)
feas, kk, ff = brute_fit(y, Fraction(10**6))
print('brute agrees:', kk == r.knots)

# 3) tiny lambda -> knots nearly everywhere (D^2 nonzero)
r = run_trend_inference(y, Fraction(1, 10**6), Fraction(1), Fraction(0), Fraction(1,20))
print('tiny lambda knots:', r.knots)
_f, kk, ff = brute_fit(y, Fraction(1, 10**6))
print('brute agrees:', kk == r.knots, 'fit equal:', ff == r.fit)

# 4) larger n stress against brute force (n=12)
rng = random.Random(9)
for n in (10, 12):
    y = [Fraction(rng.randint(-30,30)) for _ in range(n)]
    lam = Fraction(rng.randint(1,8))
    _f, kk, ff = brute_fit(y, lam)
    got = fit_trend(y, lam)
    print(f'n={n}: fit eq={ff==got.fit}, knots eq={kk==got.knots}, #knots={len(got.knots)}')

# 5) n=3 every possible sign pattern, lambda sweep
for lam in (Fraction(1,3), Fraction(7,5), Fraction(99)):
    for a in (-20, -1, 0, 7, 33):
        y = [Fraction(0), Fraction(0), Fraction(a)]
        _f, kk, ff = brute_fit(y, lam)
        got = fit_trend(y, lam)
        assert ff == got.fit and kk == got.knots, (lam, a)
print('n=3 lambda sweep OK')

# 6) region for a multi-knot case stays feasible at u=t (all random)
bad = 0
for t in range(100):
    n = rng.randint(3, 9)
    y = [Fraction(rng.randint(-25,25)) for _ in range(n)]
    lam = Fraction(rng.randint(1,6))
    r = run_trend_inference(y, lam, Fraction(1), Fraction(0), Fraction(1,20))
    if not r.changed: continue
    # t feasible
    def contains(iv, u):
        for i in iv:
            lo_ok = i.lower.__class__.__name__=='_Infinity' or (u > i.lower.value) or (u==i.lower.value and not i.lower_open)
            hi_ok = i.upper.__class__.__name__=='_Infinity' or (u < i.upper.value) or (u==i.upper.value and not i.upper_open)
            if lo_ok and hi_ok: return True
        return False
    if not contains(r.intervals, r.contrast.t): bad += 1
print('t-outside-region count:', bad)

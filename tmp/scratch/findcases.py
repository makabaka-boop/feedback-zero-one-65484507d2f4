import sys, random
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tmp/scratch')
from fractions import Fraction
from explore import brute_fit
from seg_infer.trend import run_trend_inference

rng = random.Random(555)
shown = 0
for t in range(3000):
    n = rng.randint(4, 7)
    y = [Fraction(rng.randint(-20, 20)) for _ in range(n)]
    lam = Fraction(rng.randint(1, 4))
    r = run_trend_inference(y, lam, Fraction(1), Fraction(0), Fraction(1,20))
    if not r.changed: continue
    if len(r.knots) >= 2 and len(r.intervals) >= 2:
        print('MULTI-KNOT DISCONNECTED: y=', y, 'lam=', lam, 'S=', r.knots, 'nIV=', len(r.intervals))
        for iv in r.intervals:
            lo = '-inf' if iv.lower.__class__.__name__=='_Infinity' else str(iv.lower.value)+('o' if iv.lower_open else 'c')
            hi = '+inf' if iv.upper.__class__.__name__=='_Infinity' else str(iv.upper.value)+('o' if iv.upper_open else 'c')
            print('   ', lo, hi)
        shown += 1
    if shown >= 3: break

# single knot interior sign flip (V with internal gap, open both sides)
for t in range(3000):
    n = rng.randint(4, 6)
    y = [Fraction(rng.randint(-20, 20)) for _ in range(n)]
    lam = Fraction(rng.randint(1, 4))
    r = run_trend_inference(y, lam, Fraction(1), Fraction(0), Fraction(1,20))
    if not r.changed or len(r.knots) != 1: continue
    # find a case with a bounded gap (two intervals, both finite-ish)
    if len(r.intervals) == 2 and all(iv.lower.__class__.__name__!='_Infinity' and iv.upper.__class__.__name__!='_Infinity' for iv in r.intervals):
        print('INTERNAL GAP: y=', y, 'lam=', lam, 'S=', r.knots)
        for iv in r.intervals:
            print('   ', iv.lower.value, iv.lower_open, iv.upper.value, iv.upper_open)
        break

# confirm dual sign flips across gap on the single-knot example style
r = run_trend_inference([Fraction(0),Fraction(0),Fraction(12)], Fraction(1), Fraction(1), Fraction(0), Fraction(1,20))
c = r.contrast
for u in (Fraction(-10), Fraction(0), Fraction(10)):
    yu = [c.z[i] + c.eta[i]*u/6 for i in range(3)]
    feas, kk, x = brute_fit(yu, Fraction(1))
    state0 = feas[0][0]
    print('u=', u, 'knots=', kk, 'dual state s1=', state0[0], 'x=', x)

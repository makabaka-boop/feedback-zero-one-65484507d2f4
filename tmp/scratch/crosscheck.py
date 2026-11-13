import sys, random
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tmp/scratch')
from fractions import Fraction
from explore import brute_fit
from seg_infer.trend import fit_trend, run_trend_inference, build_trend_contrast, trend_stability_region

rng = random.Random(123)
mismatch = 0
for t in range(400):
    n = rng.randint(3, 8)
    y = [Fraction(rng.randint(-20, 20), rng.randint(1, 3)) for _ in range(n)]
    lam = Fraction(rng.randint(1, 10), rng.randint(1, 3))
    feas, bknots, bx = brute_fit(y, lam)
    f = fit_trend(y, lam)
    if tuple(bx) != f.fit or tuple(bknots) != f.knots:
        mismatch += 1
        print('MISMATCH', y, lam, bx, f.fit, bknots, f.knots)
print('fit cross-check mismatches:', mismatch)

# Region grid check using independent refit at every probe
def in_region(intervals, u):
    for iv in intervals:
        lo = -10**9 if iv.lower.__class__.__name__ == '_Infinity' else float(iv.lower.value)
        hi = 10**9 if iv.upper.__class__.__name__ == '_Infinity' else float(iv.upper.value)
        if lo < float(u) < hi: return True
        if u == iv.lower.value if iv.lower.__class__.__name__ != '_Infinity' else False:
            if not iv.lower_open: return True
        if iv.upper.__class__.__name__ != '_Infinity' and u == iv.upper.value:
            if not iv.upper_open: return True
    return False

region_mismatch = 0
cases_with_flip = 0
for t in range(60):
    n = rng.randint(3, 7)
    y = [Fraction(rng.randint(-12, 12)) for _ in range(n)]
    lam = Fraction(rng.randint(1, 5))
    res = run_trend_inference(y, lam, Fraction(1), Fraction(0), Fraction(1,20))
    if not res.changed: continue
    c = res.contrast
    norm = c.eta_norm_sq
    intervals = res.intervals
    # gather endpoints
    eps = [e for iv in intervals for e in (iv.lower, iv.upper) if e.__class__.__name__ != '_Infinity']
    vals = sorted({e.value for e in eps})
    # probes: dense rational grid + endpoint-near points
    probes = [Fraction(i, 2) for i in range(-120, 121)]
    for v in vals:
        probes += [v - Fraction(1,1000), v, v + Fraction(1,1000)]
    flip_seen = False
    for u in probes:
        yu = [c.z[i] + c.eta[i]*u/norm for i in range(n)]
        _, kk, _ = brute_fit(yu, lam)
        expected = tuple(kk) == res.knots
        got = in_region(intervals, u)
        if expected != got:
            region_mismatch += 1
            print('REGION MISMATCH', t, u, expected, got, y, lam)
    # detect sign flip: evaluate dual sign of first knot across feasible probes
    signs = set()
    for u in [Fraction(i) for i in range(-80, 81)]:
        yu = [c.z[i] + c.eta[i]*u/norm for i in range(n)]
        feas, kk, xx = brute_fit(yu, lam)
        if tuple(kk) == res.knots:
            # find dual at first knot via feasible states
            j = res.knots[0]
            for state, s, x, obj in feas:
                if state[j-1] != 0:
                    signs.add(int(state[j-1]))
    if len(signs) == 2:
        cases_with_flip += 1
print('region mismatches:', region_mismatch, '| sign-flip cases seen:', cases_with_flip)

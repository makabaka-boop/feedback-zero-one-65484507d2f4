import sys, time, random
sys.path.insert(0, '/workspace')
from fractions import Fraction
from seg_infer.trend import fit_trend, run_trend_inference

rng = random.Random(3)
for n in (8, 10, 12, 14):
    y = [Fraction(rng.randint(-30,30)) for _ in range(n)]
    lam = Fraction(rng.randint(1,3))
    t0 = time.time()
    r = run_trend_inference(y, lam, Fraction(1), Fraction(0), Fraction(1,20))
    dt = time.time()-t0
    print(f'n={n} #knots={len(r.knots)} time={dt:.2f}s #intervals={len(r.intervals) if r.changed else 0}')

# worst case: tiny lambda -> many knots
n = 12
y = [Fraction(rng.randint(-30,30)) for _ in range(n)]
t0=time.time()
r = run_trend_inference(y, Fraction(1,100), Fraction(1), Fraction(0), Fraction(1,20))
print(f'tiny-lambda n={n} #knots={len(r.knots)} time={time.time()-t0:.2f}s')

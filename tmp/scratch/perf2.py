import sys, time, random
sys.path.insert(0, '/workspace')
from fractions import Fraction
from seg_infer.trend import run_trend_inference
rng = random.Random(11)
for n in (16, 20, 24):
    y = [Fraction(rng.randint(-40,40)) for _ in range(n)]
    lam = Fraction(rng.randint(1,2))
    t0 = time.time()
    r = run_trend_inference(y, lam, Fraction(1), Fraction(0), Fraction(1,20))
    print(f'n={n} #knots={len(r.knots)} time={time.time()-t0:.2f}s iv={len(r.intervals)}')

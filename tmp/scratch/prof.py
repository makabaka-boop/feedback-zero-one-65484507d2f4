import sys, cProfile, pstats, random
sys.path.insert(0, '/workspace')
from fractions import Fraction
from seg_infer.trend import run_trend_inference
rng = random.Random(3)
n = 14
y = [Fraction(rng.randint(-30,30)) for _ in range(n)]
lam = Fraction(rng.randint(1,3))
pr = cProfile.Profile()
pr.enable()
r = run_trend_inference(y, lam, Fraction(1), Fraction(0), Fraction(1,20))
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(12)

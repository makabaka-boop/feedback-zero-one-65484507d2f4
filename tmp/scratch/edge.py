import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tmp/scratch')
from fractions import Fraction
from explore import brute_fit
from seg_infer.trend import run_trend_inference
from seg_infer.input_parsing import parse_job, JobRequest
from seg_infer.engine import serialize_job_result
from seg_infer.errors import InputError

# 1) threshold equality: y2=6 => no_change; just above 6 => change with open boundary
for val in (Fraction(6), Fraction(6,1) + Fraction(1,1000)):
    r = run_trend_inference([Fraction(0),Fraction(0),val], Fraction(1), Fraction(1), Fraction(0), Fraction(1,20))
    print('val=', val, 'changed=', r.changed, 'knots=', r.knots)

# threshold boundary openness: take a changing case and inspect endpoints
r = run_trend_inference([Fraction(0),Fraction(0),Fraction(12)], Fraction(1), Fraction(1), Fraction(0), Fraction(1,20))
for iv in r.intervals:
    print('iv:', iv)

# 2) multi-knot disconnected region: pick multi-knot example and report V
y = [Fraction(15),Fraction(-11),Fraction(-3),Fraction(5),Fraction(-14),Fraction(-13)]
r = run_trend_inference(y, Fraction(1), Fraction(1), Fraction(0), Fraction(1,20))
print('multi knots:', r.knots, '# intervals:', len(r.intervals))
for iv in r.intervals:
    lo = '-inf' if iv.lower.__class__.__name__=='_Infinity' else str(iv.lower.value)
    hi = '+inf' if iv.upper.__class__.__name__=='_Infinity' else str(iv.upper.value)
    print(f'  ({lo}{"o" if iv.lower_open else "c"}, {hi}{"o" if iv.upper_open else "c"})')
print('t inside? t=', r.contrast.t)

# verify knot set exactly matches at rational endpoints with the open/closed flags
def in_region(intervals, u):
    for iv in intervals:
        if iv.lower.__class__.__name__=='_Infinity':
            lo_ok = True
        elif u == iv.lower.value:
            lo_ok = not iv.lower_open
        else:
            lo_ok = u > iv.lower.value
        if iv.upper.__class__.__name__=='_Infinity':
            hi_ok = True
        elif u == iv.upper.value:
            hi_ok = not iv.upper_open
        else:
            hi_ok = u < iv.upper.value
        if lo_ok and hi_ok: return True
    return False
c = r.contrast; norm = c.eta_norm_sq
eps = sorted({e.value for iv in r.intervals for e in (iv.lower, iv.upper) if e.__class__.__name__!='_Infinity'})
for v in eps:
    for u in (v - Fraction(1,10**6), v, v + Fraction(1,10**6)):
        yu = [c.z[i] + c.eta[i]*u/norm for i in range(len(y))]
        _, kk, _ = brute_fit(yu, Fraction(1))
        exp = tuple(kk) == r.knots
        got = in_region(r.intervals, u)
        assert exp == got, (u, exp, got)
print('endpoint neighborhood exact check OK')

# 3) input parsing validation
def bad(doc, frag):
    try:
        parse_job(doc); print('FAIL no error', doc)
    except InputError as e:
        assert frag in str(e), (str(e), frag)
        print('rejected:', e)

base_trend = {"y":[0,0,12],"mode":"trend","lambda":"1/1","s2":"1","mu0":"0","alpha":"1/20"}
bad({**base_trend, "lambda":"0/1"}, "$.lambda")
bad({**base_trend, "lambda":"-2"}, "$.lambda")
bad({**base_trend, "lambda":"1.5"}, "$.lambda")
bad({**base_trend, "y":[0,1]}, "$.y")
bad({**base_trend, "mode":"seg"}, "$.mode")
bad({**base_trend, "k":2}, "$.k")
bad({**base_trend, "lambda":True}, "$.lambda")
seg = {"y":[0,1],"k":2,"s2":"1","mu0":"0","alpha":"1/20"}
bad({**seg, "lambda":"1/1"}, "$.lambda")
bad({**seg, "k":"2"}, "$.k")
# missing lambda
doc = {"y":[0,0,1],"mode":"trend","s2":"1","mu0":"0","alpha":"1/20"}
bad(doc, "$.lambda")
# good jobs
j = parse_job(base_trend); print('trend job:', j.mode, j.penalty)
j2 = parse_job(seg); print('segment job:', j2.mode, j2.k)
out = serialize_job_result(run_trend_inference(j.y, j.penalty, j.s2, j.mu0, j.alpha))
print('serialized keys:', sorted(out.keys()))

import time, json
from hexspace import HexSpace
from bootstrap2 import Bootstrap2
t0 = time.time()
H = HexSpace('ext', verbose=False).build(12)
print('built H_12 in %.0fs' % (time.time() - t0), flush=True)
B = Bootstrap2(H)
out = {}
for L in range(1, 7):
    t = time.time()
    c, log, free = B.run(L, conditions=('collinear', 'origin', 'FF collinear', 'Euclidean MRK'), verbose=False)
    out[L] = log
    print(L, [m for _, m in log], '%.1fs' % (time.time() - t), flush=True)
json.dump(out, open('cascades6.json', 'w'), indent=1)

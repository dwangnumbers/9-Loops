"""
Hexagon symbol alphabet for the six-gluon amplitude in planar N=4 SYM,
and the pair relations (integrability + extended Steinmann) used by the bootstrap.

Letters (order = index):  a, b, c, d, e, f, yu, yv, yw
  a = u/(v w),  b = v/(w u),  c = w/(u v)
  d = (1-u)/u,  e = (1-v)/v,  f = (1-w)/w      (d,e,f are m_u, m_v, m_w in older papers)
  yu, yv, yw  : parity-odd letters (y -> 1/y under parity)
Cross ratios are rational in the y variables:
  u = yu (1-yv)(1-yw) / ((1-yu yv)(1-yu yw))   and cyclically.
"""
import itertools, random
import sympy as sp

yu, yv, yw = sp.symbols('y_u y_v y_w')
Y = (yu, yv, yw)

def cross_ratios():
    u = yu*(1-yv)*(1-yw)/((1-yu*yv)*(1-yu*yw))
    v = yv*(1-yw)*(1-yu)/((1-yv*yw)*(1-yv*yu))
    w = yw*(1-yu)*(1-yv)/((1-yw*yu)*(1-yw*yv))
    return u, v, w

NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'yu', 'yv', 'yw']
TEX = [r'\hat a', r'\hat b', r'\hat c', r'\hat d', r'\hat e', r'\hat f', 'y_u', 'y_v', 'y_w']
ODD = [0, 0, 0, 0, 0, 0, 1, 1, 1]          # parity of each letter

def letters():
    u, v, w = cross_ratios()
    L = [u/(v*w), v/(w*u), w/(u*v), (1-u)/u, (1-v)/v, (1-w)/w, yu, yv, yw]
    return [sp.factor(sp.together(x)) for x in L]

def dlog_matrix():
    """9 x 3 matrix of d log(letter) / d y_k (rational functions)."""
    L = letters()
    return [[sp.simplify(sp.diff(sp.log(x), y)) for y in Y] for x in L]

def integrability_relations(npts=12, seed=1):
    """Return a basis (list of 81-vectors over Q) of the space of linear functionals
    on adjacent pairs  sum c_{ab} x_a (x) x_b  that must vanish: the 2-form
    sum c_ab dlog x_a ^ dlog x_b = 0.  The functionals are the coefficients of
    dy_i ^ dy_j, evaluated at random rational points."""
    D = dlog_matrix()
    f = sp.lambdify(Y, D, 'sympy')
    rng = random.Random(seed)
    rows = []
    for _ in range(npts):
        pt = [sp.Rational(rng.randint(2, 97), rng.randint(2, 97)) for _ in range(3)]
        M = f(*pt)
        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            row = []
            for a in range(9):
                for b in range(9):
                    row.append(M[a][i]*M[b][j] - M[a][j]*M[b][i])
            rows.append(row)
    R = sp.Matrix(rows)
    # row space basis (exact)
    Rr, piv = R.rref()
    basis = [list(Rr.row(k)) for k in range(len(piv))]
    return basis

def steinmann_relations():
    """F^{a,b} = F^{b,c} = F^{c,a} = 0  (as functionals on the 81 pairs)."""
    rels = []
    for (x, y) in [(0, 1), (1, 2), (2, 0)]:
        r = [0]*81
        r[9*x + y] = 1
        rels.append(r)
    return rels

if __name__ == '__main__':
    for n, x in zip(NAMES, letters()):
        print(n, '=', x)
    I = integrability_relations()
    print('integrability relations:', len(I), ' -> integrable pairs:', 81 - len(I))
    M = sp.Matrix(I + steinmann_relations())
    print('integrability + (extended) Steinmann:', M.rank(), ' -> allowed pairs:', 81 - M.rank())

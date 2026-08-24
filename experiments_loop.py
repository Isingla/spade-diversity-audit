"""experiments_loop.py — closed-loop and growth experiments behind the
README's numbers. Run inside the SPADE repo root (imports
spade.core.env_memory). CPU-only.

  E1  dose-response: curriculum entropy vs initial regret advantage,
      stock vs LABEL-FREE cosine gate, 3 seeds
  E2  author-rebuttal conditions: grounding injection & self-correcting
      regret (the conditions under which stock holds)
  E3  teaching value: held-out masked-prediction error for drifted vs
      diversity-held pools
  E4  compounding addresses: commissioned fusions open new midpoints;
      frontier growth vs external-drip baseline
"""
import ast, hashlib, itertools, math, random
import numpy as np
from collections import Counter

try:
    import sys; sys.path.insert(0, ".")
    from spade.core.env_memory import EnvironmentMemory
except Exception:
    EnvironmentMemory = None

MECH = {
"stack": "class Env:\n    def reset(self):\n        self.piles=[[3,2,1],[],[]]\n    def step(self,a):\n        s,d=a\n        if self.piles[s] and (not self.piles[d] or self.piles[s][-1] < self.piles[d][-1]):\n            self.piles[d].append(self.piles[s].pop())\n        r = 1 if len(self.piles[2])==3 else 0\n        return self.piles, r, r==1\n",
"guess": "class Env:\n    def reset(self):\n        self.secret=self.rng.randint(0,99); self.tries=0\n    def step(self,a):\n        self.tries += 1\n        hint = 1 if a < self.secret else (-1 if a > self.secret else 0)\n        r = 1 if a == self.secret else 0\n        return hint, r, r==1 or self.tries>=8\n",
"walk": "class Env:\n    def reset(self):\n        self.pos=0; self.goal=7; self.adj=self.build()\n    def step(self,a):\n        self.pos = self.adj[self.pos][a % len(self.adj[self.pos])]\n        r = 1 if self.pos == self.goal else 0\n        return self.pos, r, r==1\n",
"trade": "class Env:\n    def reset(self):\n        self.gold=10; self.inv={}\n    def step(self,a):\n        item,qty=a\n        cost = self.price[item]*qty\n        if cost <= self.gold:\n            self.gold -= cost; self.inv[item]=self.inv.get(item,0)+qty\n        r = self.value() - 10\n        return self.inv, r, self.gold<=0\n",
"balance": "class Env:\n    def reset(self):\n        self.left=[]; self.right=[]; self.weights=[5,3,2,1]\n    def step(self,a):\n        w,side=a\n        (self.left if side==0 else self.right).append(self.weights[w])\n        r = 1 if sum(self.left)==sum(self.right) and self.left else 0\n        return (sum(self.left),sum(self.right)), r, r==1\n",
}

def reskin(c, i):
    for k, v in {"piles": f"cr{i}", "secret": f"gm{i}", "gold": f"cd{i}",
                 "left": f"pa{i}", "pos": f"ps{i}"}.items():
        c = c.replace(k, v)
    return c

VOC = {}
def raw(code):
    ev = Counter()
    for n in ast.walk(ast.parse(code)):
        if isinstance(n, ast.Compare):
            for op in n.ops: ev[("c", type(op).__name__)] += 1
        elif isinstance(n, ast.BinOp): ev[("b", type(n.op).__name__)] += 1
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            ev[("f", n.func.attr)] += 1
        elif isinstance(n, ast.AugAssign): ev[("a", type(n.op).__name__)] += 1
        elif isinstance(n, ast.Subscript): ev[("s", "")] += 1
        elif isinstance(n, (ast.Dict, ast.DictComp)): ev[("d", "")] += 1
        elif isinstance(n, (ast.List, ast.ListComp)): ev[("l", "")] += 1
    for k in ev: VOC.setdefault(k, len(VOC))
    return ev
for _c in MECH.values(): raw(_c)
COUPLE = [("k", f"nov{i}") for i in range(40)]
for _k in COUPLE: VOC.setdefault(_k, len(VOC))
D = 96
def vp(ev):
    v = np.zeros(D)
    for k, c in ev.items():
        if VOC[k] < D: v[VOC[k]] = c
    return v
def nv(ev):
    x = vp(ev); return x / max(x.sum(), 1)

def entropy_loop(gated, seed, adv=0.2, grounding=0.0, mastery=False, cycles=12):
    rng = random.Random(seed)
    mem = EnvironmentMemory(max_size=2000, seed=seed)
    fams = list(MECH)
    base = {f: 0.5 for f in fams}; base["stack"] = 0.5 * (1 + adv)
    exposure = Counter(); vecs = []
    for cyc in range(cycles):
        seeds_ = mem.high_regret_seeds(n=2) if mem.records else []
        boost = Counter(s.skill for s in seeds_)
        for _ in range(15):
            if rng.random() < grounding:
                f = rng.choice(fams)                       # grounding doc sets topic
            else:
                w = [base[f] * (1 + 2 * boost.get(f, 0)) for f in fams]
                f = rng.choices(fams, weights=w)[0]
            exposure[f] += 1
            r_eff = base[f] / (1 + 0.15 * exposure[f]) if mastery else base[f]
            x = vp(raw(reskin(MECH[f], rng.randrange(999))))
            xn = x / max(np.linalg.norm(x), 1e-9)
            if gated and sum(1 for v in vecs if float(v @ xn) > 0.92) >= 3:
                continue                                   # label-free cosine cap
            vecs.append(xn)
            mem.add(f"e{cyc}_{rng.randrange(10**7)}.py", f, "x",
                    rng.uniform(.2, .8), regret=r_eff * rng.uniform(0.8, 1.2))
        cnt = Counter(r.skill for r in mem.records)
        p = np.array(list(cnt.values()), float); p /= p.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(fams)))

def E1_E2():
    def cell(**kw):
        st = [entropy_loop(False, s, **kw) for s in (1, 2, 3)]
        gt = [entropy_loop(True, s, **kw) for s in (1, 2, 3)]
        return (f"stock {np.mean(st):.2f} (min {min(st):.2f}) | "
                f"gated {np.mean(gt):.2f} (min {min(gt):.2f})")
    print("E1 dose-response (initial regret advantage of one family):")
    for adv in (0.0, 0.1, 0.2, 0.4, 0.8):
        print(f"  +{int(adv*100):3d}%: {cell(adv=adv)}")
    print("E2 author-rebuttal conditions (+20% adv):")
    print(f"  grounding=0.3 : {cell(grounding=0.3)}")
    print(f"  mastery decay : {cell(mastery=True)}")
    print(f"  both          : {cell(grounding=0.6, mastery=True)}")

def E3():
    from sklearn.linear_model import Ridge
    names = list(MECH)
    def build(kind, n=30):
        rng = random.Random(1); out = []
        for i in range(n):
            f = ("stack" if rng.random() < 0.8 else rng.choice(names)) \
                if kind == "drifted" else names[i % 5]
            out.append(raw(reskin(MECH[f], i)))
        return out
    def cfuse(a, b): return a + "\n" + b.replace("class Env", "class Env2")
    tests = [raw(cfuse(MECH[a], MECH[b]))
             for a, b in list(itertools.combinations(names, 2))[5:]]
    rng = np.random.default_rng(0)
    def err(pool):
        M = np.stack([nv(e) for e in pool]); out = []
        for t in tests:
            x = nv(t)
            for _ in range(60):
                m = rng.random(D) < 0.4
                A = M.copy(); A[:, m] = 0
                w = Ridge(alpha=.1).fit(A, M[:, m])
                c = x.copy(); c[m] = 0
                out.append(float(np.abs(w.predict(c[None])[0] - x[m]).mean()))
        return float(np.mean(out))
    print("E3 teaching value (held-out error on unseen combinations; lower=better):")
    for kind in ("drifted", "diverse"):
        print(f"  {kind:8s}: {err(build(kind)):.4f}")

def E4():
    R = {f: Counter(raw(c)) for f, c in MECH.items()}
    def integrate(a, b, pid):
        ev = Counter(a) + Counter(b)
        h = int(hashlib.sha1(pid.encode()).hexdigest(), 16)
        for j in range(2): ev[COUPLE[(h + j) % 40]] = 1 + (h >> j) % 3
        return ev
    def growth(commission, cycles=15):
        concepts = {f: Counter(R[f]) for f in MECH}
        order = {f: 0 for f in MECH}
        ext = iter(["vote", "melt", "queue", "lock", "race"])
        acc = mo = a0 = a1 = 0
        for cyc in range(cycles):
            names = list(concepts)
            empty = [p for p in itertools.combinations(names, 2)
                     if "+".join(sorted(p)) not in names]
            if cyc == 0: a0 = len(empty)
            a1 = len(empty)
            if commission and empty:
                a, b = empty[cyc % len(empty)]
                ev = integrate(concepts[a], concepts[b], a + "+" + b)
                xn = nv(ev); Ea = nv(concepts[a]); Eb = nv(concepts[b])
                m = vp(concepts[a]) + vp(concepts[b]); m /= max(m.sum(), 1)
                dA = float(np.linalg.norm(xn - Ea)); dB = float(np.linalg.norm(xn - Eb))
                dm = float(np.linalg.norm(xn - m))
                if dA > 0.02 and dB > 0.02 and 0.005 < dm < min(dA, dB):
                    nm = "+".join(sorted((a, b)))
                    concepts[nm] = ev
                    order[nm] = max(order[a], order[b]) + 1
                    acc += 1; mo = max(mo, order[nm])
            elif not commission and cyc % 3 == 2:
                n = next(ext, None)
                if n:
                    concepts[n] = Counter({("l", ""): 2, ("c", "Eq"): 1, ("s", ""): 2})
                    order[n] = 0
        return len(concepts), a0, a1, acc, mo
    print("E4 compounding addresses (15 cycles):")
    for mode, label in ((False, "external drip"), (True, "commissioned ")):
        nc, a0, a1, acc, mo = growth(mode)
        print(f"  {label}: concepts 5->{nc} | addresses {a0}->{a1} | "
              f"accepted {acc} | deepest order {mo}")

if __name__ == "__main__":
    if EnvironmentMemory is None:
        print("[E1/E2 skipped — run inside the SPADE repo root]")
    else:
        E1_E2()
    E3(); E4()

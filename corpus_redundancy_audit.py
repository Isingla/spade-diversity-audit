"""corpus_redundancy_audit.py — measure concept-level redundancy of the
released SPADE environment corpus (spade-rl on Hugging Face).

Produces the headline number for the GitHub issue: what fraction of the
7,872 released environments are mechanically redundant under
concept-level identity (mechanics, not words).

Usage on a fresh pod:
    pip install datasets numpy
    python corpus_redundancy_audit.py \
        --dataset spade-rl/SPADE-Environment-Pool-GPT5.5-Games \
        --out audit_report.json

Outputs (printed + JSON):
  1. STRICT redundancy: environments whose mechanics signature is an
     EXACT duplicate of an earlier environment's (identical op multiset
     across reset/step — renamings invisible, mechanics identical).
     This is the defensible headline number.
  2. NEAR redundancy at cosine >= 0.95 and >= 0.90 over op-distribution
     vectors (report alongside, clearly labeled).
  3. Cross-skill duplicates: same mechanics signature filed under two
     different declared skills (reskins crossing category lines).
  4. Concept-cloud stats: number of distinct mechanics clusters vs pool
     size; largest cluster share (the drift measure).
  5. Vectose occupancy: mass-weighted midpoints of the top concept
     clusters — populated vs empty (the commissioning queue).
CPU-only; ~7.9k AST parses + blocked cosine, minutes on any box.
"""
import argparse, ast, hashlib, json, sys
from collections import Counter, defaultdict
import numpy as np


# ---------------- feature extraction (mechanics, not words) -----------
def op_events(code):
    """Multiset of mechanic events, keyed per containing function where
    possible (reset/step/other) so structure isn't fully flattened."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    events = Counter()

    def fn_of(node, stack):
        for s in reversed(stack):
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return s.name if s.name in ("reset", "step") else "other"
        return "module"

    stack = []

    def walk(node):
        stack.append(node)
        scope = fn_of(node, stack)
        if isinstance(node, ast.Compare):
            for op in node.ops:
                events[(scope, "cmp", type(op).__name__)] += 1
        elif isinstance(node, ast.BinOp):
            events[(scope, "bin", type(node.op).__name__)] += 1
        elif isinstance(node, ast.BoolOp):
            events[(scope, "bool", type(node.op).__name__)] += 1
        elif isinstance(node, ast.AugAssign):
            events[(scope, "aug", type(node.op).__name__)] += 1
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else None)
            if name in ("append", "pop", "get", "count", "randint", "random",
                        "choice", "shuffle", "sort", "sorted", "min", "max",
                        "sum", "len", "abs", "range", "enumerate", "zip",
                        "keys", "values", "items", "add", "remove", "insert",
                        "index", "join", "split", "setdefault", "update"):
                events[(scope, "call", name)] += 1
        elif isinstance(node, ast.Subscript):
            events[(scope, "subscript", "")] += 1
        elif isinstance(node, (ast.Dict, ast.DictComp)):
            events[(scope, "dict", "")] += 1
        elif isinstance(node, (ast.List, ast.ListComp)):
            events[(scope, "list", "")] += 1
        elif isinstance(node, (ast.Set, ast.SetComp)):
            events[(scope, "set", "")] += 1
        elif isinstance(node, ast.While):
            events[(scope, "while", "")] += 1
        elif isinstance(node, ast.For):
            events[(scope, "for", "")] += 1
        elif isinstance(node, ast.If):
            events[(scope, "if", "")] += 1
        elif isinstance(node, ast.Return):
            events[(scope, "return", "")] += 1
        for ch in ast.iter_child_nodes(node):
            walk(ch)
        stack.pop()

    walk(tree)
    return events


def sig_hash(events):
    """Exact mechanics signature: hash of the sorted event multiset."""
    key = json.dumps(sorted((list(k), v) for k, v in events.items()))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ---------------- corpus loading (schema auto-detected) ---------------
def load_corpus(dataset, split):
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    cols = ds.column_names
    # code column = string column most often containing "def step"
    code_col, best = None, -1
    for c in cols:
        v = ds[0][c]
        if isinstance(v, str):
            hits = sum(1 for i in range(min(50, len(ds)))
                       if isinstance(ds[i][c], str) and "def step" in ds[i][c])
            if hits > best:
                best, code_col = hits, c
    skill_col = next((c for c in cols if c.lower() in
                      ("skill", "skill_name", "category", "task_type")), None)
    print(f"[schema] columns={cols}\n[schema] code column -> '{code_col}' "
          f"({best}/50 contain 'def step'); skill column -> {skill_col}")
    return ds, code_col, skill_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default="spade-rl/SPADE-Environment-Pool-GPT5.5-Games")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="audit_report.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = full corpus")
    args = ap.parse_args()

    ds, code_col, skill_col = load_corpus(args.dataset, args.split)
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"[load] {n} environments")

    sigs, vecs, skills, parse_fail, vocab = [], [], [], 0, {}
    raw_events = []
    for i in range(n):
        ev = op_events(ds[i][code_col])
        if ev is None:
            parse_fail += 1
            raw_events.append(None); sigs.append(None); skills.append(None)
            continue
        raw_events.append(ev)
        sigs.append(sig_hash(ev))
        skills.append(ds[i][skill_col] if skill_col else "n/a")
        for k in ev:
            vocab.setdefault(k, len(vocab))
    print(f"[parse] {parse_fail} failures ({parse_fail/max(n,1):.1%})")

    X = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, ev in enumerate(raw_events):
        if ev is None:
            continue
        for k, v in ev.items():
            X[i, vocab[k]] = v
    mass = X.sum(1)
    ok = mass > 0
    Xn = np.zeros_like(X)
    Xn[ok] = X[ok] / mass[ok, None]

    # ---- 1. strict redundancy -----------------------------------------
    seen, strict_dup, groups = {}, 0, defaultdict(list)
    for i, s in enumerate(sigs):
        if s is None:
            continue
        groups[s].append(i)
        if s in seen:
            strict_dup += 1
        else:
            seen[s] = i
    valid = n - parse_fail
    print(f"\n[1] STRICT concept duplicates: {strict_dup}/{valid} "
          f"= {strict_dup/max(valid,1):.1%} of parsable environments share an "
          f"exact mechanics signature with an earlier one")
    big = sorted(groups.values(), key=len, reverse=True)[:5]
    print("    largest identical-mechanics groups:",
          [len(g) for g in big])

    # ---- 2. near redundancy (blocked cosine) --------------------------
    Un = Xn.copy()
    norms = np.linalg.norm(Un, axis=1); ok2 = norms > 0
    Un[ok2] = Un[ok2] / norms[ok2, None]
    near95 = np.zeros(n, bool); near90 = np.zeros(n, bool)
    B = 512
    for a in range(0, n, B):
        S = Un[a:a+B] @ Un.T
        for r in range(S.shape[0]):
            i = a + r
            S[r, i] = 0.0
            j = int(np.argmax(S[r]))
            if j < i or sigs[i] != sigs[j]:  # count vs ANY other env
                m = float(S[r].max())
                near95[i] = m >= 0.95
                near90[i] = m >= 0.90
    print(f"[2] NEAR duplicates: cosine>=0.95: {near95.sum()/max(valid,1):.1%} "
          f"| cosine>=0.90: {near90.sum()/max(valid,1):.1%} "
          f"(op-distribution; report clearly labeled, not as headline)")

    # ---- 3. cross-skill identical mechanics ---------------------------
    if skill_col:
        cross = sum(1 for g in groups.values()
                    if len({skills[i] for i in g}) > 1)
        print(f"[3] identical-mechanics groups spanning MULTIPLE declared "
              f"skills: {cross} (reskins crossing category lines)")

    # ---- 4. concept-cloud shape ---------------------------------------
    n_clusters = len(groups)
    biggest = len(big[0]) / max(valid, 1) if big else 0
    print(f"[4] distinct mechanics signatures: {n_clusters} for {valid} envs "
          f"(ratio {n_clusters/max(valid,1):.2f}); largest single concept = "
          f"{biggest:.1%} of pool")

    # ---- 5. vectose occupancy over top concepts -----------------------
    reps = [g[0] for g in big if len(g) >= 3][:12]
    occupied = empty = 0
    if len(reps) >= 2:
        R = X[reps]; m = mass[reps]
        for a in range(len(reps)):
            for b in range(a+1, len(reps)):
                mid = (R[a] + R[b]) / (m[a] + m[b])
                midn = mid / max(np.linalg.norm(mid), 1e-9)
                sims = Un @ midn
                sims[reps[a]] = sims[reps[b]] = 0
                if float(sims.max()) >= 0.97:
                    occupied += 1
                else:
                    empty += 1
        print(f"[5] vectose occupancy among top concept pairs: "
              f"{occupied} populated / {empty} EMPTY "
              f"(empty midpoints = enumerable fusion commissions)")

    json.dump({
        "n": n, "parse_failures": parse_fail,
        "strict_duplicates": strict_dup,
        "strict_rate": strict_dup / max(valid, 1),
        "near95_rate": float(near95.sum() / max(valid, 1)),
        "near90_rate": float(near90.sum() / max(valid, 1)),
        "distinct_signatures": n_clusters,
        "largest_concept_share": biggest,
        "largest_groups": [len(g) for g in big],
        "vectose_occupied": occupied, "vectose_empty": empty,
    }, open(args.out, "w"), indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()

"""corpus_neardup.py — near-twin rates over the cached SPADE corpus:
cosine similarity of op-distribution vectors at 0.90/0.95/0.98/0.99,
plus the top-scoring named pairs for manual inspection.
Run: python corpus_neardup.py  (after corpus_audit_v3.py has cached
the snapshot). Needs corpus_redundancy_audit.py in the same folder."""
import json
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download
from corpus_redundancy_audit import op_events

root = Path(snapshot_download(
    "spade-rl/SPADE-Environment-Pool-GPT5.5-Games", repo_type="dataset"))
files = sorted((root / "games").glob("*.py"))
print(f"[load] {len(files)} files; featurizing...")
vocab, rows, names = {}, [], []
for p in files:
    ev = op_events(p.read_text(encoding="utf-8", errors="ignore"))
    if ev is None: continue
    for k in ev: vocab.setdefault(k, len(vocab))
    rows.append(ev); names.append(p.name)
X = np.zeros((len(rows), len(vocab)), dtype=np.float32)
for i, ev in enumerate(rows):
    for k, v in ev.items(): X[i, vocab[k]] = v
X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
n = len(X); best = np.zeros(n); who = np.zeros(n, int); B = 256
for a in range(0, n, B):
    S = X[a:a+B] @ X.T
    for r in range(S.shape[0]):
        S[r, a+r] = -1
        who[a+r] = int(np.argmax(S[r])); best[a+r] = float(S[r].max())
    if a % 2048 == 0: print(f"  {a}/{n}")
for t in (0.99, 0.98, 0.95, 0.90):
    print(f"[near] cosine>={t}: {(best>=t).mean():.1%} of corpus has a near-twin")
for i in np.argsort(-best)[:5]:
    print(f"   {best[i]:.3f}  {names[i]}  <->  {names[who[i]]}")
json.dump({"near99": float((best>=.99).mean()), "near98": float((best>=.98).mean()),
           "near95": float((best>=.95).mean()), "near90": float((best>=.90).mean())},
          open("neardup_report.json", "w"), indent=2)
print("[out] neardup_report.json")

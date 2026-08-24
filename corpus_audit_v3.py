"""corpus_audit_v3.py — strict concept-duplicate audit of the released
SPADE corpus. Reads the cached HF snapshot's games/*.py directly.
Run: python corpus_audit_v3.py --limit 200   then full with no --limit.
Needs: pip install huggingface_hub numpy ; corpus_redundancy_audit.py
in the same folder (imports its featurizer)."""
import argparse, json
from pathlib import Path
from collections import defaultdict
from huggingface_hub import snapshot_download
from corpus_redundancy_audit import op_events, sig_hash

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="audit_report.json")
    a = ap.parse_args()
    root = Path(snapshot_download(
        "spade-rl/SPADE-Environment-Pool-GPT5.5-Games", repo_type="dataset"))
    files = sorted((root / "games").glob("*.py"))
    if a.limit: files = files[:a.limit]
    print(f"[load] {len(files)} game files")
    sigs, skills, fails = [], [], 0
    for p in files:
        code = p.read_text(encoding="utf-8", errors="ignore")
        parts = p.stem.split("_")            # game_000000_mathematical_reasoning
        skills.append("_".join(parts[2:]) if len(parts) > 2 else "n/a")
        ev = op_events(code)
        if ev is None: fails += 1; sigs.append(None); continue
        sigs.append(sig_hash(ev))
    valid = len(files) - fails
    groups = defaultdict(list)
    for i, s in enumerate(sigs):
        if s: groups[s].append(i)
    dup = sum(len(g) - 1 for g in groups.values())
    big = sorted(groups.values(), key=len, reverse=True)[:5]
    cross = sum(1 for g in groups.values()
                if len(g) > 1 and len({skills[i] for i in g}) > 1)
    print(f"\n[1] STRICT concept duplicates: {dup}/{valid} = {dup/max(valid,1):.1%}")
    print(f"[2] parse failures: {fails} ({fails/max(len(files),1):.1%})")
    print(f"[3] largest identical-mechanics groups: {[len(g) for g in big]}")
    if big:
        print(f"[4] distinct signatures: {len(groups)} for {valid} envs; "
              f"largest concept = {len(big[0])/max(valid,1):.1%} of pool; "
              f"{len(set(skills))} declared skills")
        if len(big[0]) > 1:
            print("    example group:", [files[i].name for i in big[0][:4]])
    print(f"[5] duplicate groups spanning MULTIPLE declared skills: {cross}")
    json.dump({"n": len(files), "valid": valid, "strict_duplicates": dup,
               "strict_rate": dup/max(valid,1), "parse_failures": fails,
               "distinct_signatures": len(groups),
               "largest_groups": [len(g) for g in big],
               "cross_skill_groups": cross}, open(a.out, "w"), indent=2)
    print(f"[out] {a.out}")

if __name__ == "__main__":
    main()

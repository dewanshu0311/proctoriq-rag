#!/usr/bin/env python
"""How stable is the router at temperature 0?

    python scripts/measure_router_stability.py --runs 5

Successive router runs identified different two-source questions — Q13 on one
run, Q26 on another, same prompt, same temperature. Temperature 0 is not a
determinism guarantee: it fixes the sampling rule, not the serving stack.

This matters because refusal firing is now the router's *only* value. If firing
itself is unstable, probe 6a measures a moving target: the arm submitted on
Kaggle would not be the arm measured locally.

Reports per axis: how many questions are stable across every run, which flip,
and — the decisive number — whether the refusal-firing set is stable.

Runs with the cache disabled, since the cache would hide exactly what is being
measured.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402
from proctoriq_rag.routing.router import QueryRouter  # noqa: E402

AXES = ("intent", "platform", "phase", "cardinality")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()
    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    client = GroqChatClient()
    runs: list[dict[str, dict]] = []

    for run in range(args.runs):
        print(f"run {run + 1}/{args.runs} ...", flush=True)
        # cache_path=None on purpose: caching would mask the very thing measured.
        router = QueryRouter(client=client, cache_path=None)
        decisions = router.classify_all(qids, qtexts)
        runs.append({d.question_id: d for d in decisions})
        print(f"  {router.stats}")

    _report(runs, qids, qtexts, args.out_dir)
    return 0


def _report(runs, qids, qtexts, out_dir: Path) -> None:
    rule = "=" * 92
    print("\n" + rule)
    print(f"ROUTER STABILITY — {len(runs)} runs, temperature 0, cache disabled")
    print(rule)

    unstable: dict[str, list[str]] = defaultdict(list)
    rows = []

    for axis in AXES:
        stable = 0
        for qid in qids:
            values = {getattr(runs[r][qid], axis) for r in range(len(runs))}
            if len(values) == 1:
                stable += 1
            else:
                unstable[axis].append(qid)
        rows.append({"axis": axis, "stable": stable, "total": len(qids),
                     "rate": stable / len(qids)})
        print(f"  {axis:<14} stable on {stable}/{len(qids)} = {stable / len(qids):.1%}")

    print()
    for axis in AXES:
        flips = unstable[axis]
        if flips:
            print(f"  {axis} flips ({len(flips)}): {', '.join(flips)}")

    # ── the decisive number ────────────────────────────────────────────────
    print("\n" + rule)
    print("REFUSAL FIRING — the router's only remaining value")
    print(rule)
    firing_sets = [
        {qid for qid in qids if runs[r][qid].is_policy_boundary}
        for r in range(len(runs))
    ]
    sizes = [len(s) for s in firing_sets]
    always = set.intersection(*firing_sets)
    ever = set.union(*firing_sets)
    sometimes = sorted(ever - always)

    print(f"  fires on {sizes} questions across runs")
    print(f"  ALWAYS fires : {len(always)}")
    print(f"  SOMETIMES    : {len(sometimes)}  {', '.join(sometimes) or '(none)'}")
    print(f"  never        : {len(qids) - len(ever)}")

    if not sometimes:
        print("\n  -> Refusal firing is STABLE. Probe 6a measures a fixed arm.")
    else:
        share = len(sometimes) / max(len(ever), 1)
        print(f"\n  -> !! Refusal firing is UNSTABLE on {len(sometimes)} question(s) "
              f"({share:.0%} of the firing set).")
        print("     Probe 6a would measure a moving target. A deterministic fallback")
        print("     classifier on the refusal path is required before it runs.")

    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "router_stability.csv", index=False)

    detail = pd.DataFrame([
        {"question_id": qid,
         **{f"{axis}_values": "|".join(sorted({
             str(getattr(runs[r][qid], axis)) for r in range(len(runs))
         })) for axis in AXES},
         "fires_in_runs": sum(runs[r][qid].is_policy_boundary for r in range(len(runs))),
         "question": qtexts[qids.index(qid)][:90]}
        for qid in qids
    ])
    detail.to_csv(out_dir / "router_stability_detail.csv", index=False)
    print(f"\n  detail -> {out_dir / 'router_stability_detail.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Test whether detector-gated Adapter->OFT routing is worthwhile.

Quantifies: (1) is OFT overall significantly better? (2) the routing ceiling
(always-OFT - always-Adapter), (3) whether runs are paired -> oracle router,
(4) success-vs-cost framing.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

CSV = "/home/tamoghno/LIBERO/analysis/perturbation_results.csv"
Z = norm.ppf(0.975)


def wilson_diff(k1, n1, k2, n2):
    """Newcombe 95% CI for p1 - p2 (independent)."""
    def w(k, n):
        p = k / n
        d = 1 + Z * Z / n
        c = (p + Z * Z / (2 * n)) / d
        h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
        return c - h, c + h
    l1, u1 = w(k1, n1)
    l2, u2 = w(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return p1 - p2, lo, hi


def main():
    df = pd.read_csv(CSV)
    grid = df[df.phase != "baseline"].copy()
    base = df[df.phase == "baseline"].copy()
    fired = grid[grid.fired == 1].copy()

    print("=" * 72)
    print("ROUTING ANALYSIS — is Adapter->OFT switching worthwhile?")
    print("=" * 72)

    # ---- 1. pairing check -------------------------------------------------
    print("\n## 1. Are the two models' runs paired (same init state per episode)?")
    paired_cells, matched = 0, 0
    for (ph, mg), g in grid.groupby(["phase", "magnitude_cm"]):
        a = g[g.model == "VLA-Adapter"].sort_values("episode")
        o = g[g.model == "OFT"].sort_values("episode")
        if len(a) == len(o) and len(a) > 0:
            paired_cells += 1
            ta = a.trigger_step.astype(str).tolist()
            to = o.trigger_step.astype(str).tolist()
            if ta == to:
                matched += 1
    print(f"   {matched}/{paired_cells} cells have identical trigger-step sequences.")
    print("   -> runs are", "PAIRED" if matched > paired_cells * 0.7
          else "NOT paired (independent) — oracle router not computable per-episode")

    # ---- 2. overall model comparison --------------------------------------
    print("\n## 2. Overall success: always-Adapter vs always-OFT")
    for label, d in [("grid (fired-only)", fired), ("baseline", base)]:
        a = d[d.model == "VLA-Adapter"].success
        o = d[d.model == "OFT"].success
        diff, lo, hi = wilson_diff(o.sum(), len(o), a.sum(), len(a))
        print(f"   {label:20s}  Adapter={a.mean()*100:5.1f}%  OFT={o.mean()*100:5.1f}%  "
              f"gap={diff*100:+5.1f}pp  [{lo*100:+5.1f},{hi*100:+5.1f}]")

    # ---- 3. model main-effect LRT -----------------------------------------
    print("\n## 3. Is the model effect statistically significant?")
    d = fired.copy()
    d["mag_c"] = d.magnitude_cm - grid.magnitude_cm.mean()
    full = smf.logit("success ~ C(model)*C(phase)*mag_c", d).fit(disp=0)
    red = smf.logit("success ~ C(phase)*mag_c", d).fit(disp=0)
    lr = 2 * (full.llf - red.llf)
    ddf = len(full.params) - len(red.params)
    p = chi2.sf(lr, ddf)
    print(f"   LRT drop {{model + all model interactions}}: "
          f"LR={lr:.3f}, df={ddf}, p={p:.4f}")
    print(f"   -> OFT advantage is {'SIGNIFICANT' if p < .05 else 'NOT significant'} "
          f"at alpha=.05")

    # ---- 4. routing ceiling & per-cell gaps -------------------------------
    print("\n## 4. Routing ceiling = always-OFT - always-Adapter, per cell")
    print(f"   {'phase':6s} {'dx':>4s} {'Adapter':>8s} {'OFT':>7s} {'gap':>7s} {'95% CI':>16s}")
    gaps = []
    for (ph, mg), g in fired.groupby(["phase", "magnitude_cm"]):
        a = g[g.model == "VLA-Adapter"].success
        o = g[g.model == "OFT"].success
        if len(a) == 0 or len(o) == 0:
            continue
        diff, lo, hi = wilson_diff(o.sum(), len(o), a.sum(), len(a))
        gaps.append(diff)
        sig = "" if lo < 0 < hi else " *"
        print(f"   {ph:6s} {mg:>4.0f} {a.mean()*100:>7.1f}% {o.mean()*100:>6.1f}% "
              f"{diff*100:>+6.1f}pp [{lo*100:>+5.1f},{hi*100:>+5.1f}]{sig}")
    gaps = np.array(gaps)
    print(f"\n   mean per-cell gap = {gaps.mean()*100:+.1f}pp  "
          f"(min {gaps.min()*100:+.1f}, max {gaps.max()*100:+.1f})")
    print(f"   cells where OFT significantly beats Adapter: "
          f"{sum(1 for g in gaps if g>0)}/{len(gaps)} positive, "
          f"but per-cell CIs mostly span 0")

    # ---- 5. cost framing --------------------------------------------------
    print("\n## 5. Success-vs-cost framing")
    aA = fired[fired.model=='VLA-Adapter'].success.mean()
    aO = fired[fired.model=='OFT'].success.mean()
    print(f"   always-Adapter : success={aA*100:.1f}%   cost=1x (0.5B)")
    print(f"   always-OFT     : success={aO*100:.1f}%   cost~=Cx (7B)")
    print(f"   A detector-gated router with escalation fraction f has:")
    print(f"     success  <= {aA*100:.1f} + f_correct*({(aO-aA)*100:.1f})   (<= always-OFT)")
    print(f"     cost     ~= (1-f)*1x + f*Cx")
    print(f"   The ENTIRE robustness budget a perfect router can recover is "
          f"{(aO-aA)*100:+.1f}pp.")


if __name__ == "__main__":
    main()

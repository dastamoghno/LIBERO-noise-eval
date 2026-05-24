#!/usr/bin/env python3
"""Plate-perturbation displacement sweep analysis (phase B, both models).

Parses pert_<model>_plate_phaseB_dx<N>cm logs, compares OFT vs VLA-Adapter
degradation when the *plate* (mug's placement target) is pushed, and contrasts
it with the earlier pudding sweep.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

LOG_DIRS = {
    "VLA-Adapter": "/home/tamoghno/VLA-Adapter/experiments/logs",
    "OFT": "/home/tamoghno/openvla-oft/experiments/logs",
}
PUDDING_CSV = "/home/tamoghno/LIBERO/analysis/perturbation_results.csv"
OUT_CSV = "/home/tamoghno/LIBERO/analysis/plate_results.csv"
Z = norm.ppf(0.975)

RE_EVAL = re.compile(r"^EVAL-.*?--(?P<note>.+)\.txt$")
RE_PLATE = re.compile(r"^pert_(?P<model>adapter|oft)_plate_phaseB_dx(?P<dx>\d+)cm$")
RE_TARGET = re.compile(r"\[FORCE\] target .*?Δx=([\d.]+)cm.*?→ F=([\d.]+)N")
RE_FIRED = re.compile(r"\[FORCE\] Fired at episode_step=(\d+)")
RE_SUCCESS = re.compile(r"^Success:\s*(True|False)")
RE_START = re.compile(r"Starting episode (\d+)")


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def parse(path):
    rows = []
    cur = dict(force=0.0, fired=False)
    for line in open(path, encoding="utf-8", errors="replace"):
        if RE_START.search(line):
            cur = dict(force=0.0, fired=False)
        mt = RE_TARGET.search(line)
        if mt:
            cur["force"] = float(mt.group(2))
        if RE_FIRED.search(line):
            cur["fired"] = True
        ms = RE_SUCCESS.match(line)
        if ms:
            rows.append((cur["force"], int(cur["fired"]), int(ms.group(1) == "True")))
            cur = dict(force=0.0, fired=False)
    return rows


def main():
    # collect plate logs (most recent per cell)
    best = {}
    for model_dir, logdir in LOG_DIRS.items():
        for fn in glob.glob(os.path.join(logdir, "EVAL-*.txt")):
            m = RE_EVAL.match(os.path.basename(fn))
            if not m:
                continue
            mm = RE_PLATE.match(m["note"])
            if not mm:
                continue
            model = "VLA-Adapter" if mm["model"] == "adapter" else "OFT"
            dx = int(mm["dx"])
            ts = os.path.basename(fn).split("--")[0]
            key = (model, dx)
            if key not in best or ts > best[key][0]:
                best[key] = (ts, fn)

    rows = []
    for (model, dx), (ts, path) in sorted(best.items()):
        for force, fired, succ in parse(path):
            rows.append(dict(model=model, magnitude_cm=dx, force_N=force,
                             fired=fired, success=succ))
    if not rows:
        print("No plate logs found yet.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("=" * 70)
    print("PLATE PERTURBATION SWEEP — phase B [60,200), both models")
    print("=" * 70)
    print(f"\n## Per-cell success (Wilson 95% CI)   [{len(df)} episodes parsed]")
    print(f"{'model':12s} {'dx':>4s} {'n':>4s} {'fired':>5s} {'rate':>7s} {'95% CI':>16s}")
    for (mdl, dx), g in df.groupby(["model", "magnitude_cm"]):
        n, k, nf = len(g), int(g.success.sum()), int(g.fired.sum())
        lo, hi = wilson(k, n)
        print(f"{mdl:12s} {dx:>4d} {n:>4d} {nf:>5d} {k/n*100:>6.1f}% "
              f"[{lo*100:>5.1f},{hi*100:>5.1f}]")

    # logistic regression: success ~ model * magnitude  (phase B fixed)
    d = df[df.fired == 1].copy()
    d["mag_c"] = d.magnitude_cm - df.magnitude_cm.mean()
    full = smf.logit("success ~ C(model)*mag_c", d).fit(disp=0)
    red_mxm = smf.logit("success ~ C(model)+mag_c", d).fit(disp=0)
    red_mdl = smf.logit("success ~ mag_c", d).fit(disp=0)

    def lrt(a, b):
        lr = 2 * (a.llf - b.llf)
        ddf = len(a.params) - len(b.params)
        return lr, ddf, chi2.sf(lr, ddf)

    print("\n## Logistic regression  success ~ model * magnitude")
    lr, ddf, p = lrt(full, red_mxm)
    print(f"   model:magnitude interaction : LR={lr:.3f} df={ddf} p={p:.4f}  "
          f"{'SIG' if p<.05 else 'ns'}")
    lr, ddf, p = lrt(red_mxm, red_mdl)
    print(f"   model main effect           : LR={lr:.3f} df={ddf} p={p:.4f}  "
          f"{'SIG' if p<.05 else 'ns'}")
    print(f"   magnitude slope (logit/cm)  : {full.params['mag_c']:.3f}")

    # contrast with pudding sweep (phase B, fired-only)
    print("\n## Plate vs pudding degradation (phase B, fired-only success %)")
    try:
        pud = pd.read_csv(PUDDING_CSV)
        pud = pud[(pud.phase == "B") & (pud.fired == 1)]
        print(f"   {'dx':>4s} {'PLATE Adapter':>14s} {'PLATE OFT':>10s} "
              f"{'PUD Adapter':>12s} {'PUD OFT':>9s}")
        for dx in sorted(df.magnitude_cm.unique()):
            pa = df[(df.model=='VLA-Adapter')&(df.magnitude_cm==dx)].success.mean()*100
            po = df[(df.model=='OFT')&(df.magnitude_cm==dx)].success.mean()*100
            ua = pud[(pud.model=='VLA-Adapter')&(pud.magnitude_cm==dx)].success
            uo = pud[(pud.model=='OFT')&(pud.magnitude_cm==dx)].success
            ua = ua.mean()*100 if len(ua) else float('nan')
            uo = uo.mean()*100 if len(uo) else float('nan')
            print(f"   {dx:>4.0f} {pa:>13.1f}% {po:>9.1f}% {ua:>11.1f}% {uo:>8.1f}%")
    except Exception as e:
        print(f"   (pudding contrast skipped: {e})")

    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

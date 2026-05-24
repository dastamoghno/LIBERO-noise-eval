#!/usr/bin/env python3
"""Statistical analysis: does perturbation phase / magnitude significantly
distinguish OpenVLA-OFT from VLA-Adapter robustness?

Logistic regression  success ~ model * phase * magnitude  + likelihood-ratio
tests for each interaction, per-cell Wilson CIs, simulation-based power, plots.

Run in the `pertstats` conda env:
  /home/tamoghno/miniconda3/envs/pertstats/bin/python perturbation_stats.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

CSV = "/home/tamoghno/LIBERO/analysis/perturbation_results.csv"
PLOTS = "/home/tamoghno/LIBERO/analysis/plots"
REPORT = "/home/tamoghno/LIBERO/analysis/stats_report.txt"
Z = norm.ppf(0.975)

_out = []
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    _out.append(line)


def wilson(k, n, z=Z):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def lrt(full, reduced):
    """LR test of nested logits: 2*Δllf ~ chi2(Δparams)."""
    lr = 2.0 * (full.llf - reduced.llf)
    ddf = len(full.params) - len(reduced.params)
    return lr, ddf, chi2.sf(lr, ddf)


def holm(pvals):
    """Holm step-down adjusted p-values, returned in input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def fit(formula, data):
    return smf.logit(formula, data=data).fit(disp=0, maxiter=200)


def main():
    df = pd.read_csv(CSV)
    emit("=" * 78)
    emit("PERTURBATION ROBUSTNESS ANALYSIS  —  OFT vs VLA-Adapter, LIBERO-10 task 6")
    emit("=" * 78)

    base = df[df.phase == "baseline"].copy()
    grid = df[df.phase != "baseline"].copy()

    # ---- descriptives: per-cell Wilson CIs ---------------------------------
    emit("\n## Per-cell success rates (Wilson 95% CI)")
    emit(f"{'model':12s} {'phase':6s} {'dx':>4s} {'n':>4s} {'fired':>5s} "
         f"{'rate':>7s} {'95% CI':>16s}")
    cell_tab = []
    for (mdl, ph, dx), g in grid.groupby(["model", "phase", "magnitude_cm"]):
        n, k = len(g), int(g.success.sum())
        nf = int(g.fired.sum())
        lo, hi = wilson(k, n)
        emit(f"{mdl:12s} {ph:6s} {dx:>4.0f} {n:>4d} {nf:>5d} "
             f"{k/n*100:>6.1f}% [{lo*100:>5.1f},{hi*100:>5.1f}]")
        cell_tab.append(dict(model=mdl, phase=ph, dx=dx, n=n, fired=nf,
                             rate=k / n, lo=lo, hi=hi))
    for (mdl,), g in base.groupby(["model"]):
        n, k = len(g), int(g.success.sum())
        lo, hi = wilson(k, n)
        emit(f"{mdl:12s} {'BASE':6s} {0:>4d} {n:>4d} {'-':>5s} "
             f"{k/n*100:>6.1f}% [{lo*100:>5.1f},{hi*100:>5.1f}]")
    cell_df = pd.DataFrame(cell_tab)

    # ---- fired confound: primary = fired-only ------------------------------
    n_total = len(grid)
    grid_fired = grid[grid.fired == 1].copy()
    emit(f"\n## `fired` confound: {len(grid_fired)}/{n_total} grid episodes "
         f"actually received the push.")
    for ph, g in grid.groupby("phase"):
        emit(f"   phase {ph}: fired {int(g.fired.sum())}/{len(g)} "
             f"({g.fired.mean()*100:.0f}%)")
    emit("   Primary analysis uses fired-only; robustness refit adds C(fired).")

    # ---- model prep --------------------------------------------------------
    def prep(d):
        d = d.copy()
        d["mag_c"] = d.magnitude_cm - grid.magnitude_cm.mean()
        return d
    D = prep(grid_fired)
    emit(f"\n   magnitude centered at {grid.magnitude_cm.mean():.3f} cm")

    REF = "C(model, Treatment('VLA-Adapter')) * C(phase, Treatment('A')) * mag_c"
    FULL = f"success ~ {REF}"

    # ---- full logistic regression ------------------------------------------
    emit("\n## Full logistic regression:  success ~ model * phase * magnitude")
    try:
        full = fit(FULL, D)
        emit(full.summary().as_text())
    except Exception as e:
        emit(f"   full fit failed ({e}); retrying L1-penalized")
        full = smf.logit(FULL, data=D).fit_regularized(method="l1", alpha=0.01, disp=0)
        emit("   (L1-penalized fit used — separation detected)")

    cond = np.linalg.cond(full.model.exog)
    emit(f"\n   design-matrix condition number = {cond:.1f} "
         f"({'ok' if cond < 30 else 'elevated — check collinearity'})")

    # ---- likelihood-ratio tests for interactions ---------------------------
    emit("\n## Likelihood-ratio tests for interaction terms (hierarchical)")
    M = "C(model, Treatment('VLA-Adapter'))"
    P = "C(phase, Treatment('A'))"
    all2 = f"success ~ ({M} + {P} + mag_c)**2"
    red_mxmag = f"success ~ {M}*{P} + mag_c + {P}:mag_c"   # drop model:mag
    red_mxph  = f"success ~ {M}*mag_c + {P}*mag_c"          # drop model:phase

    m_all2 = fit(all2, D)
    m_mxmag = fit(red_mxmag, D)
    m_mxph = fit(red_mxph, D)

    tests = []
    lr, ddf, p = lrt(full, m_all2)
    tests.append(("model:phase:magnitude (3-way)", lr, ddf, p))
    lr, ddf, p = lrt(m_all2, m_mxmag)
    tests.append(("model:magnitude", lr, ddf, p))
    lr, ddf, p = lrt(m_all2, m_mxph)
    tests.append(("model:phase", lr, ddf, p))

    praw = np.array([t[3] for t in tests])
    padj = holm(praw)
    emit(f"\n{'interaction':32s} {'LR chi2':>9s} {'df':>3s} {'p_raw':>9s} {'p_Holm':>9s}  sig")
    for (name, lr, ddf, p), pa in zip(tests, padj):
        star = "***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else "ns"
        emit(f"{name:32s} {lr:>9.3f} {ddf:>3d} {p:>9.4f} {pa:>9.4f}  {star}")
    emit("\n   model:magnitude  -> does perturbation MAGNITUDE distinguish the models?")
    emit("   model:phase      -> does perturbation PHASE distinguishes the models?")

    # ---- robustness refit with fired covariate (all rows) ------------------
    emit("\n## Robustness: refit on ALL grid rows with C(fired) covariate")
    Dall = prep(grid)
    try:
        rob = fit(f"success ~ {REF} + C(fired)", Dall)
        rob2 = fit(all2 + " + C(fired)", Dall)
        for name, fullm, redm in [
            ("model:phase:magnitude (3-way)", rob, rob2)]:
            lr, ddf, p = lrt(fullm, redm)
            emit(f"   {name}: LR={lr:.3f} df={ddf} p={p:.4f}  "
                 f"(fired-only gave p={tests[0][3]:.4f})")
        emit("   -> conclusions stable if signs/significance match fired-only.")
    except Exception as e:
        emit(f"   robustness refit skipped ({e})")

    # ---- simulation-based power --------------------------------------------
    emit("\n## Simulation-based power at n=50/cell (parametric bootstrap, B=800)")
    emit("   power to detect each interaction AT THE OBSERVED effect size:")
    rng = np.random.default_rng(7)
    B = 800
    mu = np.clip(full.predict(D).to_numpy(), 1e-6, 1 - 1e-6)
    power = {name: 0 for name, *_ in tests}
    valid = 0
    for _ in range(B):
        sim = D.copy()
        sim["success"] = rng.binomial(1, mu)
        try:
            sf = fit(FULL, sim)
            sa = fit(all2, sim)
            sm = fit(red_mxmag, sim)
            sp = fit(red_mxph, sim)
        except Exception:
            continue
        valid += 1
        if lrt(sf, sa)[2] < 0.05:
            power["model:phase:magnitude (3-way)"] += 1
        if lrt(sa, sm)[2] < 0.05:
            power["model:magnitude"] += 1
        if lrt(sa, sp)[2] < 0.05:
            power["model:phase"] += 1
    emit(f"   ({valid}/{B} sims converged)")
    for name in power:
        emit(f"   {name:32s} power = {power[name]/max(valid,1)*100:5.1f}%")
    emit("   (power <80% => a non-significant result is inconclusive, not 'no effect')")

    # ---- plots -------------------------------------------------------------
    _plots(D, full, cell_df, base)
    emit(f"\n## Plots written to {PLOTS}/")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(_out) + "\n")
    emit(f"## Full report saved to {REPORT}")


def _plots(D, full, cell_df, base):
    phases = ["A", "B", "C"]
    pname = {"A": "pre-mug", "B": "mug-carry/return", "C": "pudding-interaction"}
    models = ["VLA-Adapter", "OFT"]
    colors = {"VLA-Adapter": "#d6604d", "OFT": "#4393c3"}
    magmean = D.magnitude_cm.mean() if len(D) else 4.25
    # use the grid-wide mean already baked into mag_c: recompute reference
    # (mag_c = magnitude_cm - grid.mean); reconstruct grid mean from D
    grid_mean = (D.magnitude_cm - D.mag_c).iloc[0] if len(D) else 4.25

    # (a) success vs magnitude, panel per phase
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    grid_dx = np.linspace(1, 8, 50)
    for ax, ph in zip(axes, phases):
        for mdl in models:
            sub = cell_df[(cell_df.model == mdl) & (cell_df.phase == ph)]
            if len(sub):
                ax.errorbar(sub.dx, sub.rate * 100,
                            yerr=[(sub.rate - sub.lo) * 100, (sub.hi - sub.rate) * 100],
                            fmt="o", color=colors[mdl], capsize=3, label=f"{mdl} (obs)")
            pred = pd.DataFrame(dict(model=mdl, phase=ph,
                                     mag_c=grid_dx - grid_mean))
            try:
                yhat = full.predict(pred) * 100
                ax.plot(grid_dx, yhat, "-", color=colors[mdl], alpha=.8)
            except Exception:
                pass
        for mdl in models:
            b = base[base.model == mdl]
            if len(b):
                ax.axhline(b.success.mean() * 100, ls=":", color=colors[mdl], alpha=.6)
        ax.set_title(f"phase {ph}: {pname[ph]}")
        ax.set_xlabel("target displacement (cm)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("success rate (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Success vs perturbation magnitude  (points=obs ±95% CI, "
                 "lines=logistic fit, dotted=no-perturbation baseline)")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/success_vs_magnitude.png", dpi=130)
    plt.close(fig)

    # (b) interaction-coefficient forest plot
    fig, ax = plt.subplots(figsize=(8, 5))
    params = full.params
    ci = full.conf_int()
    inter = [n for n in params.index if ":" in n]
    y = np.arange(len(inter))
    ax.errorbar(params[inter], y,
                xerr=[params[inter] - ci.loc[inter, 0], ci.loc[inter, 1] - params[inter]],
                fmt="o", color="#333", capsize=3)
    ax.axvline(0, ls="--", color="red", alpha=.6)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("C(model, Treatment('VLA-Adapter'))", "model")
                         .replace("C(phase, Treatment('A'))", "phase")
                        for n in inter], fontsize=8)
    ax.set_xlabel("logit coefficient (95% Wald CI)")
    ax.set_title("Interaction coefficients")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/interaction_coefficients.png", dpi=130)
    plt.close(fig)

    # (c) phase x model heatmap of mean success
    fig, ax = plt.subplots(figsize=(6, 3.6))
    mat = (cell_df.groupby(["model", "phase"]).rate.mean()
           .unstack("phase").reindex(index=models, columns=phases))
    im = ax.imshow(mat.values * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([pname[p] for p in phases])
    ax.set_yticks(range(2)); ax.set_yticklabels(models)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{mat.values[i,j]*100:.0f}%", ha="center", va="center")
    ax.set_title("Mean success (averaged over magnitude)")
    fig.colorbar(im, label="success %")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/phase_model_heatmap.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

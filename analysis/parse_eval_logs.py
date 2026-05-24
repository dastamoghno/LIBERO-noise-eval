#!/usr/bin/env python3
"""Parse LIBERO EVAL-*.txt logs into a tidy per-episode CSV for the
OFT-vs-VLA-Adapter perturbation robustness analysis.

Selects only the factorial-grid runs:
  - new grid runs   : run_id_note = pert_<model>_baseline
                                    pert_<model>_phase{A,B,C}_dx{N}cm
  - reused phase-B  : run_id_note = [oft_]dx{1,3,5}cm_horiz_tilt0.1_trig60to200_n50

For duplicate (model, phase, magnitude) cells the most recent log wins
(EVAL filenames embed a sortable timestamp).

Output columns:
  model, phase, magnitude_cm, force_N, episode, trigger_step, fired, success
"""
import csv
import os
import re
import sys

LOG_DIRS = {
    "VLA-Adapter": "/home/tamoghno/VLA-Adapter/experiments/logs",
    "OFT": "/home/tamoghno/openvla-oft/experiments/logs",
}
OUT_CSV = "/home/tamoghno/LIBERO/analysis/perturbation_results.csv"

# run_id_note is everything after the last "--" in the EVAL filename (sans .txt)
RE_EVAL = re.compile(r"^EVAL-.*?--(?P<note>.+)\.txt$")
RE_PERT_GRID = re.compile(r"^pert_(?P<model>adapter|oft)_phase(?P<phase>[ABC])_dx(?P<dx>\d+)cm$")
RE_PERT_BASE = re.compile(r"^pert_(?P<model>adapter|oft)_baseline$")
RE_REUSED_B = re.compile(r"^(?:oft_)?dx(?P<dx>[135])cm_horiz_tilt0\.1_trig60to200_n50$")

RE_TARGET = re.compile(r"\[FORCE\] target .*?Δx=([\d.]+)cm.*?→ F=([\d.]+)N")
RE_TRIGGER = re.compile(r"\[FORCE\] Pre-sampled trigger steps: \[([\d,\s]*)\]")
RE_FIRED = re.compile(r"\[FORCE\] Fired at episode_step=(\d+)")
RE_SUCCESS = re.compile(r"^Success:\s*(True|False)")
RE_START = re.compile(r"Starting episode (\d+)")


def classify(note):
    """Map a run_id_note -> (model, phase, magnitude_cm) or None to skip."""
    m = RE_PERT_GRID.match(note)
    if m:
        model = "VLA-Adapter" if m["model"] == "adapter" else "OFT"
        return model, m["phase"], int(m["dx"])
    m = RE_PERT_BASE.match(note)
    if m:
        model = "VLA-Adapter" if m["model"] == "adapter" else "OFT"
        return model, "baseline", 0
    m = RE_REUSED_B.match(note)
    if m:
        return None, "B", int(m["dx"])  # model resolved from directory
    return None


def parse_episodes(path):
    """Yield (episode, magnitude_cm, force_N, trigger_step, fired, success)."""
    cur = dict(ep=None, mag=0.0, force=0.0, trig="", fired=False)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ms = RE_START.search(line)
            if ms:
                cur = dict(ep=int(ms.group(1)), mag=0.0, force=0.0, trig="", fired=False)
                continue
            mt = RE_TARGET.search(line)
            if mt:
                cur["mag"] = float(mt.group(1))
                cur["force"] = float(mt.group(2))
                continue
            mg = RE_TRIGGER.search(line)
            if mg:
                cur["trig"] = mg.group(1).strip()
                continue
            mf = RE_FIRED.search(line)
            if mf:
                cur["fired"] = True
                continue
            msc = RE_SUCCESS.match(line)
            if msc:
                trig = cur["trig"].split(",")[0].strip() if cur["trig"] else ""
                yield (
                    cur["ep"],
                    cur["mag"],
                    cur["force"],
                    int(trig) if trig else "",
                    cur["fired"],
                    msc.group(1) == "True",
                )
                # reset per-episode fields; keep nothing
                cur = dict(ep=None, mag=0.0, force=0.0, trig="", fired=False)


def main():
    # Collect candidate logs: {(model, phase, magnitude): (timestamp_str, path)}
    best = {}
    for model_dir, logdir in LOG_DIRS.items():
        if not os.path.isdir(logdir):
            print(f"WARN: missing {logdir}", file=sys.stderr)
            continue
        for fn in os.listdir(logdir):
            m = RE_EVAL.match(fn)
            if not m:
                continue
            cls = classify(m["note"])
            if cls is None:
                continue
            model, phase, mag = cls
            if model is None:
                model = model_dir  # reused phase-B: directory is authoritative
            ts = fn.split("--")[0]  # EVAL-...-<timestamp>
            key = (model, phase, mag)
            if key not in best or ts > best[key][0]:
                best[key] = (ts, os.path.join(logdir, fn))

    rows = []
    for (model, phase, mag), (ts, path) in sorted(best.items()):
        eps = list(parse_episodes(path))
        for ep, mag_cm, force_n, trig, fired, success in eps:
            rows.append(dict(
                model=model, phase=phase, magnitude_cm=mag_cm if phase != "baseline" else 0,
                force_N=force_n, episode=ep, trigger_step=trig,
                fired=int(fired), success=int(success),
            ))
        print(f"{model:11s} phase={phase:8s} dx={mag:>2}cm  "
              f"n={len(eps):3d}  succ={sum(e[5] for e in eps):3d}  "
              f"<- {os.path.basename(path)}")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "model", "phase", "magnitude_cm", "force_N",
            "episode", "trigger_step", "fired", "success"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}")
    # quick coverage report
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["phase"], r["magnitude_cm"]), []).append(r["success"])
    print(f"cells: {len(cells)}  (expect 24 grid + 2 baseline = 26)")
    for k in sorted(cells):
        v = cells[k]
        print(f"  {k[0]:11s} {k[1]:8s} {k[2]:>2}cm  n={len(v):3d}  rate={sum(v)/len(v)*100:5.1f}%")


if __name__ == "__main__":
    main()

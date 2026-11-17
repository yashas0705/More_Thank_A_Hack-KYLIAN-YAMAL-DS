"""compare_versions.py -- compare your my_v*.csv files with NO ground truth.

Run from the repo root:
    python compare_versions.py            summary of every evaluation/results/my_v*.csv
    python compare_versions.py --score    ...and also run the validator and the public scoreboard

For each file it prints the label shares, possessions and dead-ball stretches per half, and for
every pair of files it lists the biggest stretches where they disagree (as video times mm:ss), so
you can WATCH those moments and judge which version is more sensible. Watching is allowed; do not
turn what you see into labels or frame-specific rules. The output describes only your own files:
keep it on your machine.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks

FPS = 25
N = GAMES[GAME]["n_video_frames"]
MIN_WINDOW = 2 * FPS          # list disagreements lasting at least 2 s
TOP = 8                       # how many to list per pair


def mmss(frame):
    s = int(frame) // FPS
    return "%d:%02d" % (s // 60, s % 60)


def runs(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]))


def load(path):
    lab = pd.read_csv(path)["label"].astype(str).to_numpy()
    if len(lab) != N:
        raise SystemExit("%s has %d rows, expected %d" % (path, len(lab), N))
    return lab


def possession_lengths(seg):
    """Seconds of each run of H or of A (D and X end a run)."""
    out, n = [], 0
    for i, lab in enumerate(seg):
        if lab in ("H", "A") and i > 0 and seg[i - 1] == lab:
            n += 1
        else:
            if n:
                out.append(n / FPS)
            n = 1 if lab in ("H", "A") else 0
    if n:
        out.append(n / FPS)
    return out


def summarise(name, lab, blocks):
    print("\n=== %s ===" % name)
    for half, first, last in blocks:
        seg = lab[first:last + 1]
        share = {k: 100 * np.mean(seg == k) for k in "HAD"}
        poss = possession_lengths(seg)
        dead = [(b - a + 1) / FPS for a, b in runs(seg == "D")]
        print("half %d: H %.1f%%  A %.1f%%  D %.1f%%  X %.1f%%" % (
            half, share["H"], share["A"], share["D"], 100 * np.mean(seg == "X")))
        print("        %d possessions, mean %.1f s, median %.1f s"
              % (len(poss), np.mean(poss) if poss else 0, np.median(poss) if poss else 0))
        print("        %d dead-ball stretches, mean %.1f s, longest %.1f s"
              % (len(dead), np.mean(dead) if dead else 0, max(dead) if dead else 0))


def compare(na, a, nb, b, graded):
    print("\n=== %s vs %s ===" % (na, nb))
    differ = (a != b) & graded
    print("graded frames where they differ: %.1f%%" % (100 * differ[graded].mean()))
    for x, y, nx, ny in ((a, b, na, nb), (b, a, nb, na)):
        only = graded & (x == "D") & np.isin(y, ["H", "A"])
        print("   dead in %s but alive in %s: %d frames (%.1f%% of graded)"
              % (nx, ny, only.sum(), 100 * only[graded].mean()))
    wins = [(a_, b_) for a_, b_ in runs(differ) if b_ - a_ + 1 >= MIN_WINDOW]
    wins.sort(key=lambda w: w[1] - w[0], reverse=True)
    print("longest stretches of disagreement (watch these in the video):")
    for s, e in wins[:TOP]:
        ma = pd.Series(a[s:e + 1]).mode()[0]
        mb = pd.Series(b[s:e + 1]).mode()[0]
        print("   %s - %s (%.1f s): %s says %s, %s says %s"
              % (mmss(s), mmss(e), (e - s + 1) / FPS, na, ma, nb, mb))


def main():
    files = sorted(glob.glob(os.path.join("evaluation", "results", "my_v*.csv")))
    if not files:
        raise SystemExit("no evaluation/results/my_v*.csv files found; run the my_v*.py scripts first")
    blocks = half_video_blocks(GAME)
    graded = np.zeros(N, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    labs = {os.path.basename(f)[:-4]: load(f) for f in files}
    for name, lab in labs.items():
        summarise(name, lab, blocks)
    names = list(labs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            compare(names[i], labs[names[i]], names[j], labs[names[j]], graded)

    if "--score" in sys.argv:
        for f in files:
            print("\n--- validate %s ---" % f)
            subprocess.run([sys.executable, os.path.join("evaluation", "validate_submission.py"), f])
        print("\n--- public scoreboard (pipeline check only; never choose a version by it) ---")
        subprocess.run([sys.executable, os.path.join("evaluation", "run_evaluation.py")])


if __name__ == "__main__":
    main()

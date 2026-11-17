"""investigate_dead.py -- look at what dead balls look like in YOUR copy of the data.

Run from the repo root:   python investigate_dead.py
Everything it writes goes to data/analysis/ (git-ignored). Never commit or share those files.
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw into PNG files, no window needed
import matplotlib.pyplot as plt

from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require
from my_v3 import load_tracking            # the loader you already ran; it keeps the speeds

FPS = 25
N = GAMES[GAME]["n_video_frames"]
L, W = GAMES[GAME]["pitch"]                # pitch length and width in metres
OUT = os.path.join("data", "analysis")
os.makedirs(OUT, exist_ok=True)
LOG = []

ROWS = ["team speed (m/s)", "ball speed (m/s)", "ball visible (1=yes)",
        "ball to touchline (m)", "ball to goal line (m)",
        "ball to nearest corner (m)", "ball to centre spot (m)"]
GROUPS = [("throw-in", "throw"), ("corner", "corner"), ("goal kick", "goal kick"),
          ("free kick", "free kick"), ("kick off", "kick off")]


def say(text=""):
    print(text)
    LOG.append(text)


def fmt(frame):
    s = int(frame) // FPS
    return "%d:%02d" % (s // 60, s % 60)


def med(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a)


# ------------------------------------------------------------------ load
def get_arrays():
    cache = os.path.join(OUT, "cache.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        return z["pspeed"], z["ball"], z["bspeed"]
    (raw,) = require(["atd_raw"])
    _, pspeed, ball, bspeed = load_tracking(raw)
    np.savez(cache, pspeed=pspeed, ball=ball, bspeed=bspeed)
    return pspeed, ball, bspeed


def build_signals(pspeed, ball, bspeed):
    x = ball[:, 0] * L
    y = ball[:, 1] * W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        team_speed = np.nanmean(pspeed, axis=1)       # average speed of all tracked players
    touch = np.minimum(y, W - y)
    goal = np.minimum(x, L - x)
    corner = np.minimum.reduce([np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)])
    return {
        "team speed (m/s)": team_speed,
        "ball speed (m/s)": bspeed,
        "ball visible (1=yes)": (~np.isnan(ball[:, 0])).astype(np.float32),
        "ball to touchline (m)": touch,
        "ball to goal line (m)": goal,
        "ball to nearest corner (m)": corner,
        "ball to centre spot (m)": np.hypot(x - L / 2, y - W / 2),
        "ball to nearest pitch line (m)": np.minimum(touch, goal),
    }


def graded_mask():
    m = np.zeros(N, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        m[first:last + 1] = True
    return m


def get_tags():
    from kloppy import sportscode
    (path,) = require(["smart_tagging"])
    df = sportscode.load(path).to_df()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in ("timestamp", "end_timestamp", "code") if c not in df.columns]
    if missing:
        raise SystemExit("Smart Tagging table has no %s. Its columns are: %s" % (missing, list(df.columns)))

    def frames(col):
        if pd.api.types.is_timedelta64_dtype(col):
            secs = col.dt.total_seconds()
        else:
            secs = pd.to_numeric(col)
        return (secs * FPS).round().astype(int)

    out = pd.DataFrame({
        "start": frames(df["timestamp"]),
        "end": frames(df["end_timestamp"]),
        "code": df["code"].astype(str).str.upper(),
        "type": df["type"].fillna("").astype(str).str.lower() if "type" in df.columns else "",
        "team": df["team"].fillna("").astype(str) if "team" in df.columns else "",
    })
    return out.sort_values("start").reset_index(drop=True)


# ------------------------------------------------------------------ helpers
def runs(mask):
    """[(first, last)] of every run of True in a boolean array."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


def stack(signal, anchors, before, after):
    """One row per event, one column per frame from anchor-before to anchor+after."""
    offs = np.arange(-before, after + 1)
    mat = np.full((len(anchors), len(offs)), np.nan, dtype=np.float32)
    for r, a in enumerate(anchors):
        idx = a + offs
        ok = (idx >= 0) & (idx < len(signal))
        mat[r, ok] = signal[idx[ok]]
    return offs, mat


# ------------------------------------------------------------------ set pieces
def plot_group(name, starts, ends, sig, filename):
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(13, 2.1 * len(ROWS)), sharex="col")
    views = [(starts, 10 * FPS, 20 * FPS, "aligned on tag START (0 = start)"),
             (ends, 20 * FPS, 10 * FPS, "aligned on tag END (0 = end)")]
    for i, nm in enumerate(ROWS):
        for j, (anchors, before, after, title) in enumerate(views):
            offs, mat = stack(sig[nm], anchors, before, after)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if nm.startswith("ball visible"):
                    mid, lo, hi = np.nanmean(mat, axis=0), None, None
                else:
                    mid = np.nanmedian(mat, axis=0)
                    lo = np.nanpercentile(mat, 25, axis=0)
                    hi = np.nanpercentile(mat, 75, axis=0)
            ax = axes[i, j]
            ax.plot(offs / FPS, mid)
            if lo is not None:
                ax.fill_between(offs / FPS, lo, hi, alpha=0.25)
            ax.axvline(0, linestyle="--", linewidth=0.8)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(nm, fontsize=7)
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    fig.suptitle("%s  (%d tags; line = typical value, band = middle half of tags)" % (name, len(starts)))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=100)
    plt.close(fig)


def summarise(name, starts, ends, sig):
    say("")
    say("=== %s: %d tags ===" % (name, len(starts)))
    dur = (ends - starts) / FPS
    say("tag length in seconds: median %.1f, shortest %.1f, longest %.1f"
        % (np.median(dur), dur.min(), dur.max()))
    vis = sig["ball visible (1=yes)"]
    inside = np.concatenate([vis[s:e + 1] for s, e in zip(starts, ends)])
    say("ball visible inside the tags: %.0f%% of frames" % (100 * inside.mean()))
    say("ball position, median metres, at tag START and tag END:")
    for nm in ("ball to touchline (m)", "ball to goal line (m)",
               "ball to nearest corner (m)", "ball to centre spot (m)"):
        say("   %-30s start %6.1f   end %6.1f" % (nm, med(sig[nm][starts]), med(sig[nm][ends])))
    offs = (-10, -6, -4, -2, 0, 2, 4, 6, 10)
    say("typical values at these seconds relative to the tag START:")
    say("   %-24s " % "seconds" + " ".join("%6d" % o for o in offs))
    for nm in ("team speed (m/s)", "ball speed (m/s)", "ball visible (1=yes)"):
        vals = []
        for o in offs:
            v = sig[nm][np.clip(starts + o * FPS, 0, N - 1)]
            vals.append(np.nanmean(v) if nm.startswith("ball visible") else med(v))
        say("   %-24s " % nm + " ".join("%6.2f" % v for v in vals))
    pick = np.unique(np.linspace(0, len(starts) - 1, min(6, len(starts))).round().astype(int))
    say("moments worth watching in the video (video time mm:ss):")
    for i in pick:
        say("   %s: tag runs %s -> %s" % (name, fmt(starts[i]), fmt(ends[i])))


# ------------------------------------------------------------------ public ground truth
def plot_public_dead(found, sig, sstart, send):
    if not found:
        return
    cols = ["team speed (m/s)", "ball speed (m/s)", "ball visible (1=yes)",
            "ball to nearest pitch line (m)"]
    fig, axes = plt.subplots(len(found), len(cols), figsize=(15, 2.0 * len(found)), squeeze=False)
    for i, (a, b) in enumerate(found):
        lo, hi = max(a - 10 * FPS, 0), min(b + 10 * FPS, N - 1)
        t = (np.arange(lo, hi + 1) - a) / FPS
        for j, nm in enumerate(cols):
            ax = axes[i, j]
            ax.plot(t, sig[nm][lo:hi + 1], linewidth=0.8)
            ax.axvspan(0, (b - a) / FPS, alpha=0.2)                    # shaded = ground-truth D
            for s_ in sstart[(sstart >= lo) & (sstart <= hi)]:
                ax.axvline((s_ - a) / FPS, linestyle=":", linewidth=0.9)   # dotted = tag start
            for e_ in send[(send >= lo) & (send <= hi)]:
                ax.axvline((e_ - a) / FPS, linestyle="--", linewidth=0.9)  # dashed = tag end
            if i == 0:
                ax.set_title(nm, fontsize=8)
    fig.suptitle("public ground truth: shaded = D, dotted = set-piece tag start, dashed = tag end; "
                 "0 = start of D, seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "public_dead_stretches.png"), dpi=100)
    plt.close(fig)


def study_public_dead(sig, sp):
    path = os.path.join("evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    lab = pd.read_csv(path)["label"].astype(str).to_numpy()
    found = runs(lab == "D")
    say("")
    say("=== public ground truth: %d dead-ball (D) stretches ===" % len(found))
    sstart, send, stype = sp["start"].to_numpy(), sp["end"].to_numpy(), sp["type"].to_numpy()
    vis, team = sig["ball visible (1=yes)"], sig["team speed (m/s)"]
    pline = sig["ball to nearest pitch line (m)"]
    for k, (a, b) in enumerate(found, 1):
        before = lab[a - 1] if a > 0 else "-"
        after = lab[b + 1] if b + 1 < len(lab) else "-"
        text = ("D #%d at video time %s, %.1f s long | label before: %s, after: %s | ball visible %.0f%% | "
                "median team speed %.2f m/s | ball to nearest pitch line %.1f m at start, %.1f m at end"
                % (k, fmt(a), (b - a + 1) / FPS, before, after, 100 * vis[a:b + 1].mean(),
                   med(team[a:b + 1]), pline[a], pline[b]))
        if len(sstart):
            i = int(np.argmin(np.abs(sstart - a)))
            text += " | nearest set-piece tag: %s, starts %+.1f s from the D start" % (stype[i], (sstart[i] - a) / FPS)
        say(text)
    plot_public_dead(found, sig, sstart, send)


# ------------------------------------------------------------------ second-half start
def look_at_second_half_start(sig):
    b2 = [blk for blk in half_video_blocks(GAME) if blk[0] == 2][0][1]
    a2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    say("")
    say("=== second half start ===")
    say("graded second-half block starts at %s; the games.py kick-off anchor is %s (%.0f s later)"
        % (fmt(b2), fmt(a2), (a2 - b2) / FPS))
    lo, hi = max(b2 - 10 * FPS, 0), min(a2 + 60 * FPS, N - 1)
    names = ["team speed (m/s)", "ball speed (m/s)", "ball visible (1=yes)", "ball to centre spot (m)"]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.2 * len(names)), sharex=True)
    t = (np.arange(lo, hi + 1) - b2) / FPS
    for ax, nm in zip(axes, names):
        ax.plot(t, sig[nm][lo:hi + 1], linewidth=0.7)
        ax.set_ylabel(nm, fontsize=7)
        ax.axvline(0, linestyle="--")
        ax.axvline((a2 - b2) / FPS, linestyle="--")
    axes[-1].set_xlabel("seconds from the start of the graded second-half block (dashed: block start, kick-off anchor)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "second_half_start.png"), dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    pspeed, ball, bspeed = get_arrays()
    sig = build_signals(pspeed, ball, bspeed)
    graded = graded_mask()
    say("ball visible on graded frames, whole match: %.1f%%  (docs say 59.3%%)"
        % (100 * sig["ball visible (1=yes)"][graded].mean()))

    tags = get_tags()
    say("Smart Tagging: %d tags; codes: %s" % (len(tags), sorted(tags["code"].unique())))
    keep = (tags["code"].str.contains("SET PIECE").to_numpy()
            & graded[np.clip(tags["start"].to_numpy(), 0, N - 1)])
    sp = tags[keep].reset_index(drop=True)
    say("set-piece tags inside the graded halves: %d" % len(sp))

    ko = sp[sp["type"].str.contains("kick off", regex=False)]
    if len(ko):
        anchor = anchor_video_frame(GAMES[GAME]["kickoff"])
        say("KICK-OFF CHECK: first kick-off tag starts at %s; the games.py first-half kickoff is %s (%.1f s apart)"
            % (fmt(ko["start"].iloc[0]), fmt(anchor), (ko["start"].iloc[0] - anchor) / FPS))

    for name, key in GROUPS:
        g = sp[sp["type"].str.contains(key, regex=False)]
        if g.empty:
            say("")
            say("no '%s' tags found (look at the 'type' column of the tag table)" % name)
            continue
        starts = np.clip(g["start"].to_numpy(), 0, N - 1)
        ends = np.clip(np.maximum(g["end"].to_numpy(), g["start"].to_numpy()), 0, N - 1)
        summarise(name, starts, ends, sig)
        plot_group(name, starts, ends, sig, "group_%s.png" % name.replace(" ", "_"))

    study_public_dead(sig, sp)
    look_at_second_half_start(sig)

    say("")
    say("Done. Plots and report.txt are in %s (git-ignored: do not commit or share)." % OUT)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

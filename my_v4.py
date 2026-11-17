"""My possession detector (v4): v1's possession logic + a much stronger dead-ball detector.

Put this file in the repo root (next to games.py) and run:

    python my_v4.py

It writes evaluation/results/my_v4.csv. Then check it:

    python evaluation/validate_submission.py evaluation/results/my_v4.csv
    python evaluation/run_evaluation.py        (public 5 min: pipeline check ONLY, never for tuning)

Dead-ball evidence -- any ONE of these marks a frame dead. All are general football rules, none is
tied to a frame of this match:
  1. the ball is outside the pitch lines                                        (from v1)
  2. the ball has hardly moved for 4 s                                          (from v1)
  3. the average player speed stays at walking pace for 4 s and the ball is not travelling
  4. a Smart Tagging set piece: a window around the tag START, different for each type
  5. before the kick-off of a half (Smart Tagging "Kick Off (Start)")

Needs kloppy (pip install -r requirements-starter.txt) to read the Smart Tagging file.
"""
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks, require

# ---------------------------------------------------------------- knobs (football reasoning)
# Do NOT tune these against the public 5 minutes (RULES.md / EVALUATION.md say why).
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]     # metres: 105.3 x 68.0
N_FRAMES = GAMES[GAME]["n_video_frames"]    # 199319
FPS = 25

MAX_BALL_GAP = 25        # fill ball gaps up to 1 s by a straight line; longer gaps stay missing
CONTROL_RADIUS_M = 3.0   # a player this close to the ball can be "in control"
MARGIN_M = 1.0           # ...and must be this much closer than the other team's nearest player
SWITCH_FRAMES = 50       # the other team must look in control for ~2 s before possession flips
STILL_FRAMES = 100       # ball moved less than STILL_METRES over 4 s => stoppage
STILL_METRES = 1.0
MIN_DEAD_FRAMES = 50     # ignore dead-ball blips shorter than 2 s (they would split possessions)
CLOSE_GAP_FRAMES = 75    # a "live" blip shorter than 3 s between two stoppages is the same stoppage

# --- v4 dead-ball knobs
MAX_PLAYER_SPEED = 12.0  # m/s; faster readings are tracking errors
MIN_SPEED_COUNT = 8      # need at least this many valid player speeds in a frame
QUIET_MPS = 1.3          # team-average speed at or below this = walking pace
QUIET_WINDOW = 100       # ... held over a 4 s window
BALL_HALF_WINDOW = 25    # look 1 s back and 1 s forward from each frame
BALL_FAST_METRES = 6.0   # ball moved more than this in those 2 s => it is travelling, play is on
OUT_MIN_FRAMES = 5       # a ball outside the lines for at least 0.2 s counts as "went out"
OUT_LOOKBACK = 25 * FPS  # a ball-out up to 25 s before a set piece starts the stoppage
# seconds (before, after) the tag START that are dead. Read off the typical pattern of many tags
# of each type (team slows ~2-10 s before the start, restart ~3-12 s after), not any single moment.
TAG_WINDOWS = {"throw": (2, 6), "corner": (8, 3), "goal kick": (5, 6),
               "free kick": (6, 8), "kick off": (10, 3)}
KICKOFF_DEAD_AFTER_START_S = 3   # the ball is kicked about 3 s after a kick-off tag starts

OUT_CSV = os.path.join("evaluation", "results", "my_v4.csv")


# ---------------------------------------------------------------- 1. read the files
def _tag(el):
    return el.tag.split("}")[-1]        # drop any XML namespace


def load_team_slots(xml_path):
    """Which of the 46 track slots belong to Barcelona (home) and Athletic (away)."""
    root = ET.parse(xml_path).getroot()
    home_id = None
    for el in root.iter():
        if _tag(el) == "Team" and "barcelona" in " ".join(el.itertext()).lower():
            home_id = el.get("id")
    if home_id is None:
        raise ValueError("could not find the Barcelona team in the metadata XML")
    home, away = [], []
    for el in root.iter():
        if _tag(el) == "Player":
            slot = int(el.get("id")) - 1                  # slot index 0..45
            (home if el.get("teamId") == home_id else away).append(slot)
    assert sorted(home + away) == list(range(46)), "expected slots 1..46 exactly once"
    return np.array(sorted(home)), np.array(sorted(away))


def load_tracking(txt_path):
    """Raw tracking text -> positions, player speeds, ball, all indexed by VIDEO frame.

    Each line is   frame_id : 46 x "x,y,speed" separated by ";" : ball "x,y,speed"
    and video_frame = frame_id - 1 (DATA.md). Missing values are NaN.
    """
    players = np.full((N_FRAMES, 46, 2), np.nan, dtype=np.float32)
    speed = np.full((N_FRAMES, 46), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_id, player_part, ball_part = line.split(":")[:3]
            vf = int(frame_id) - 1
            if not 0 <= vf < N_FRAMES:
                continue
            p = np.array(player_part.replace(";", ",").split(","), dtype=np.float32)
            assert p.size == 46 * 3, "line for frame %s has %d numbers, expected 138" % (frame_id, p.size)
            p = p.reshape(46, 3)
            players[vf] = p[:, :2]
            speed[vf] = p[:, 2]
            ball[vf] = np.array(ball_part.split(";")[0].split(","), dtype=np.float32)[:2]
    return players, speed, ball


def load_set_piece_tags():
    """Smart Tagging set pieces as a table with columns: start (video frame), type (lower case)."""
    from kloppy import sportscode
    (path,) = require(["smart_tagging"])
    df = sportscode.load(path).to_df()
    df.columns = [str(c).lower() for c in df.columns]
    col = df["timestamp"]
    secs = col.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(col) else pd.to_numeric(col)
    typ = df["type"].fillna("").astype(str).str.lower() if "type" in df.columns \
        else pd.Series([""] * len(df), index=df.index)
    tags = pd.DataFrame({"start": (secs * FPS).round().astype(int),
                         "code": df["code"].astype(str).str.upper(),
                         "type": typ})
    tags = tags[tags["code"].str.contains("SET PIECE")]
    return tags.sort_values("start").reset_index(drop=True)


# ---------------------------------------------------------------- 2. clean / measure
def fill_short_gaps(xy, max_gap):
    """Straight-line fill of ball gaps up to max_gap frames. Longer gaps are left as NaN."""
    out = xy.copy()
    missing = np.isnan(xy[:, 0])
    t, n = 0, len(xy)
    while t < n:
        if not missing[t]:
            t += 1
            continue
        start = t
        while t < n and missing[t]:
            t += 1
        end = t                                            # first frame with the ball again
        if start > 0 and end < n and end - start <= max_gap:
            idx = np.arange(start, end)
            for k in (0, 1):
                out[start:end, k] = np.interp(idx, [start - 1, end], [xy[start - 1, k], xy[end, k]])
    return out


def nearest_distance(players_m, ball_m, slots):
    """Distance in metres from the ball to the closest player among `slots` (NaN if unknown)."""
    d = np.hypot(players_m[:, slots, 0] - ball_m[:, None, 0],
                 players_m[:, slots, 1] - ball_m[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.min(axis=1)
    nearest[np.isinf(nearest)] = np.nan
    return nearest


def remove_short_runs(mask, min_len):
    """Turn True-runs shorter than min_len into False."""
    mask = mask.copy()
    t, n = 0, len(mask)
    while t < n:
        if not mask[t]:
            t += 1
            continue
        start = t
        while t < n and mask[t]:
            t += 1
        if t - start < min_len:
            mask[start:t] = False
    return mask


def rolling_nanmean(values, window):
    """Centered moving average that ignores NaN. Returns (means, how many values were valid)."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    kernel = np.ones(window, dtype=np.float32)
    sums = np.convolve(np.where(valid, values, 0.0).astype(np.float32), kernel, mode="same")
    counts = np.convolve(valid.astype(np.float32), kernel, mode="same")
    means = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts


def team_speed(speed):
    """Average speed (m/s) of all tracked players in each frame; NaN if too few are tracked."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(speed) & (speed >= 0) & (speed <= MAX_PLAYER_SPEED)
    count = ok.sum(axis=1)
    total = np.where(ok, speed, 0.0).sum(axis=1)
    out = np.full(len(speed), np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count >= MIN_SPEED_COUNT)
    return out


# ---------------------------------------------------------------- 3. dead ball
def tag_kind(tag_type):
    for key in TAG_WINDOWS:
        if key in tag_type:
            return key
    return None


def tag_windows(tags, went_out, n):
    """Dead windows around set-piece tags. If the ball went out shortly before the tag, the
    stoppage is taken to start at that moment."""
    mask = np.zeros(n, dtype=bool)
    for s, typ in zip(tags["start"].to_numpy(), tags["type"].to_numpy()):
        kind = tag_kind(typ)
        if kind is None:
            continue
        before, after = TAG_WINDOWS[kind]
        lo = max(int(s) - before * FPS, 0)
        hi = min(int(s) + after * FPS, n - 1)
        base = max(int(s) - OUT_LOOKBACK, 0)
        look = went_out[base:int(s) + 1]
        if look.any():
            j = int(np.flatnonzero(look)[-1])               # the most recent ball-out ...
            while j > 0 and look[j - 1]:
                j -= 1                                      # ... back to where that run began
            lo = min(lo, base + j)
        if lo <= hi:
            mask[lo:hi + 1] = True
    return mask


def pre_kickoff(tags, blocks, n):
    """Frames of a half that come before its kick-off are dead."""
    mask = np.zeros(n, dtype=bool)
    is_start = tags["type"].str.contains("kick off", regex=False) & tags["type"].str.contains("start", regex=False)
    starts = tags.loc[is_start, "start"].to_numpy()
    for _, first, _last in blocks:
        near = starts[(starts >= first - 20 * FPS) & (starts <= first + 300 * FPS)]
        if near.size:
            s = int(near[np.argmin(np.abs(near - first))])
            end = s + KICKOFF_DEAD_AFTER_START_S * FPS
            if end > first:
                mask[first:min(end, n)] = True
    return mask


def find_dead_ball(ball_norm, ball_m, speed, tags, blocks):
    """Returns (dead mask, dict of the separate evidence masks, for the printout)."""
    n = len(ball_m)
    h = BALL_HALF_WINDOW
    with np.errstate(invalid="ignore"):
        outside = ((ball_norm[:, 0] < 0) | (ball_norm[:, 0] > 1) |
                   (ball_norm[:, 1] < 0) | (ball_norm[:, 1] > 1))
        moved = np.hypot(ball_m[STILL_FRAMES:, 0] - ball_m[:-STILL_FRAMES, 0],
                         ball_m[STILL_FRAMES:, 1] - ball_m[:-STILL_FRAMES, 1])
        still = np.zeros(n, dtype=bool)
        still[STILL_FRAMES:] = moved < STILL_METRES        # window (t-100 .. t) was still

        moved2 = np.full(n, np.nan, dtype=np.float32)      # how far the ball went in 2 s around t
        moved2[h:n - h] = np.hypot(ball_m[2 * h:, 0] - ball_m[:-2 * h, 0],
                                   ball_m[2 * h:, 1] - ball_m[:-2 * h, 1])
        travelling = moved2 > BALL_FAST_METRES             # NaN (unknown) counts as not travelling

        mean_speed, count = rolling_nanmean(team_speed(speed), QUIET_WINDOW)
        quiet = (count >= QUIET_WINDOW // 2) & (mean_speed <= QUIET_MPS)

    # a still window marks all the frames inside it as dead
    c = np.concatenate([[0], np.cumsum(still)])
    t = np.arange(n)
    hi = np.minimum(t + STILL_FRAMES + 1, n)
    still_cover = (c[hi] - c[t]) > 0

    quiet_dead = quiet & ~travelling
    tag_mask = tag_windows(tags, remove_short_runs(outside, OUT_MIN_FRAMES), n)
    pre_mask = pre_kickoff(tags, blocks, n)

    dead = outside | still_cover | quiet_dead | tag_mask | pre_mask
    dead = ~remove_short_runs(~dead, CLOSE_GAP_FRAMES)     # bridge tiny live gaps inside a stoppage
    dead = remove_short_runs(dead, MIN_DEAD_FRAMES)         # drop tiny dead blips inside play
    parts = {"ball outside lines": outside, "ball still 4 s": still_cover,
             "team at walking pace": quiet_dead, "set-piece tag window": tag_mask,
             "before kick-off": pre_mask}
    return dead, parts


# ---------------------------------------------------------------- 4. the state machine (as v1)
def label_half(d_home, d_away, dead, first, last):
    """Return the list of labels 'H'|'A'|'D'|'X' for one half (inclusive frame range)."""
    labels = ["X"] * (last - first + 1)
    state, challenger, streak_start = None, 0, 0
    for i, t in enumerate(range(first, last + 1)):
        if dead[t]:
            labels[i] = "D"
            state, challenger = None, 0      # after a stoppage there is a fresh start
            continue
        dh, da = d_home[t], d_away[t]
        if dh != dh or da != da:             # NaN: ball not seen -> change nothing
            evidence = "missing"
        elif dh < CONTROL_RADIUS_M and dh + MARGIN_M < da:
            evidence = "H"
        elif da < CONTROL_RADIUS_M and da + MARGIN_M < dh:
            evidence = "A"
        else:
            evidence = "loose"               # nobody clearly in control

        if evidence == "missing":
            pass
        elif state is None:
            if evidence in ("H", "A"):
                state = evidence
        elif evidence == state:
            challenger = 0
        elif evidence == "loose":
            challenger = max(0, challenger - 1)
        else:                                # the other team looks in control
            if challenger == 0:
                streak_start = i                 # remember when this streak began
            challenger += 1
            if challenger >= SWITCH_FRAMES:
                for j in range(streak_start, i): # backdate: possession changed when the streak began
                    labels[j] = evidence
                state, challenger = evidence, 0
        labels[i] = state if state else "X"

    # frames with no state yet (before the first clear evidence) take the label that follows
    nxt = None
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] in ("H", "A"):
            nxt = labels[i]
        elif labels[i] == "D":
            nxt = None
        elif nxt:
            labels[i] = nxt
    prev = None                              # anything left takes the label before it
    for i in range(len(labels)):
        if labels[i] == "X" and prev:
            labels[i] = prev
        elif labels[i] != "X":
            prev = labels[i]
    return labels


# ---------------------------------------------------------------- 5. report (sanity checks)
def spells(labels):
    """Lengths (frames) of possessions: runs of H or of A. D (and X) end a run."""
    out, cur, n = [], None, 0
    for lab in labels:
        if lab == cur and lab in ("H", "A"):
            n += 1
        else:
            if cur in ("H", "A"):
                out.append(n)
            cur, n = lab, 1
    if cur in ("H", "A"):
        out.append(n)
    return out


def main():
    t0 = time.time()
    meta, raw = require(["atd_meta", "atd_raw"])
    home, away = load_team_slots(meta)
    print("slots: home=%d away=%d" % (len(home), len(away)))

    print("reading tracking (this takes a minute)...")
    players, speed, ball = load_tracking(raw)
    tags = load_set_piece_tags()
    kinds = pd.Series([tag_kind(t) or "other" for t in tags["type"]]).value_counts().to_dict()
    print("set-piece tags: %d %s" % (len(tags), kinds))

    blocks = half_video_blocks(GAME)
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    # sanity numbers the docs publish: ball visible ~59.3% of graded frames, ~20.5 players tracked
    print("ball visible on graded frames: %.1f%%  (docs: 59.3%%)"
          % (100 * np.mean(~np.isnan(ball[graded, 0]))))
    print("mean tracked slots on graded frames: %.1f  (docs: 20.5)"
          % np.mean((~np.isnan(players[graded, :, 0])).sum(axis=1)))

    ball_norm = fill_short_gaps(ball, MAX_BALL_GAP)       # still 0-1 coordinates
    scale = np.array([PITCH_L, PITCH_W], dtype=np.float32)
    ball_m = ball_norm * scale                            # metres
    players_m = players * scale

    d_home = nearest_distance(players_m, ball_m, home).tolist()
    d_away = nearest_distance(players_m, ball_m, away).tolist()
    dead, parts = find_dead_ball(ball_norm, ball_m, speed, tags, blocks)

    print("dead-ball evidence, share of graded frames (rules overlap):")
    for name, mask in parts.items():
        print("   %-24s %5.1f%%" % (name, 100 * mask[graded].mean()))
    print("   %-24s %5.1f%%" % ("DEAD after cleanup", 100 * dead[graded].mean()))

    all_labels = ["X"] * N_FRAMES
    for half, first, last in blocks:
        part = label_half(d_home, d_away, dead, first, last)
        all_labels[first:last + 1] = part
        s = spells(part)
        print("half %d: %d possessions, mean %.1fs, dead-ball share %.1f%%" % (
            half, len(s), np.mean(s) / FPS if s else 0, 100 * part.count("D") / len(part)))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        f.write("frame,label\n")
        f.write("\n".join("%d,%s" % (i, lab) for i, lab in enumerate(all_labels)))
        f.write("\n")
    print("wrote %s (%d rows) in %.0fs" % (OUT_CSV, N_FRAMES, time.time() - t0))


if __name__ == "__main__":
    main()

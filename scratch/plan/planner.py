"""Whole-song planner: section record + per-bar audio intensity + target level -> per-4-bar-window generation targets.

ROUND 2 (PLAN-2-6617): the section contrast is now *paired* (every label relative to the verse of the same chart) instead of two
independent cross-chart medians; see scratch/plan/fit_targets2.py for the derivation and out/plan/targets2.json for the numbers.
Round 1 (independent p50 per label, `Targets`) is kept as `plan_v1()` and is used when a `Targets` instance is passed explicitly.

Every target value is a *fitted* quantity from the 791 official MiLK+ charts (versionid >= 10, MASTER / Re:MASTER, lv >= 11):
  mu[label, metric](level)  median over charts of  value(label mask) / value(verse mask)   (shrunk to the level group, interpolated over bands)
  level[metric](level)      chart-wide mean of the metric (all bars with notes)
  window value              = level * mu[label(w)] / Z,   Z = note-weighted mean of mu over the windows of the song  (song mean preserved)
  window density            = song mean * mu_npb[label] * exp(b_between * dLoudness_label + s_label * dLoudness_window)  (renormalised to the level density)
Outputs per window the absolute targets (`slide_share`, ...), the relative multipliers `mult` (target / song mean: what the generator multiplies
its own level-calibrated share targets with) and the creativity knobs.

  .venv\\Scripts\\python.exe scratch\\plan\\planner.py --sid 41 --slot 5 --level 14.7 [--sections out/plan/e25/Apollo.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scratch", "plan"))
import planfeat as PF  # noqa: E402

LABELS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "break", "build-up", "drop", "outro", "other")
BANDS = ("12", "12+", "13", "13+", "14", "14+")
WIN = 4
PLAN_EDGES = (0.35, 0.6, 0.8, 0.95, 1.1, 1.25, 1.5, 2.0)      # same bucket edges as mai/v3.PLAN_EDGES
CHORUS_IDS = (LABELS.index("chorus"), LABELS.index("drop"))
VERSE_ID = LABELS.index("verse")
HALF_CALM = tuple(LABELS.index(x) for x in ("intro", "pre-chorus", "bridge", "break", "outro", "other"))

# metrics copied to every window target (cmet.py definitions)
RATE_KEYS = ("slide_share", "touch_share", "hold_share", "brk_rate", "ex_rate", "each_rate", "simul_rate", "rhythm_fine",
             "rare_slide", "vrare_slide", "conn_slide", "multi_arm", "stream", "trill", "stair", "wide", "jack",
             "special_bar", "nov_lane", "motif_half", "repeat_prev", "slide_ent", "rhythm_ent", "lane_ent", "npb_p90")
# shares / rates that are normalised so that the song mean stays at the chart-wide level (A-mask) of the band
NORM_KEYS = ("slide_share", "touch_share", "hold_share", "brk_rate", "ex_rate", "each_rate", "simul_rate", "rhythm_fine",
             "rare_slide", "vrare_slide", "conn_slide", "multi_arm", "stream", "trill", "stair", "wide", "jack", "special_bar")
# the multipliers the generator consumes (planner JSON `mult`)
MULT_KEYS = NORM_KEYS


def band_of(lv):
    """cmet.band_of, with a fallback to the nearest band for levels below 12"""
    lv = float(lv)
    if lv < 12.5:
        return "12"
    if lv < 13.0:
        return "12+"
    if lv < 13.5:
        return "13"
    if lv < 14.0:
        return "13+"
    if lv < 14.5:
        return "14"
    return "14+"


def lv_center(band):
    return {"12": 12.25, "12+": 12.75, "13": 13.25, "13+": 13.75, "14": 14.25, "14+": 14.7}[band]


# ------------------------------------------------------------------------------------------------ round-1 tables (kept)
class Targets:
    """round 1: independent per-label p50 tables (out/plan/targets.json).  Kept for reproducing the round-1 report."""

    def __init__(self, path=None):
        p = path or os.path.join(ROOT, "out", "plan", "targets.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        self.d = d
        self.bands = d["bands"]
        self.elas = d["elasticity"]

    def value(self, label, metric, level):
        xs, ys = [], []
        for b in BANDS:
            cell = self.bands[b].get(label, {}).get(metric)
            if cell and cell.get("n", 0) >= 12:
                xs.append(lv_center(b))
                ys.append(cell["p50"])
        if not xs:
            cell = self.bands["all"].get(label, {}).get(metric)
            return cell["p50"] if cell else float("nan")
        v = float(np.interp(float(level), xs, ys))
        return float(np.clip(v, min(ys), max(ys)))

    def elasticity(self, label, group):
        e = self.elas.get(group, {}).get(label)
        if e is None:
            e = self.elas.get("all", {}).get(label)
        return float(e["slope"]) if e else 0.0

    def label_table(self, label, level, group):
        return {k: self.value(label, k, level) for k in ("npb",) + RATE_KEYS}


# ------------------------------------------------------------------------------------------------ round-2 tables
class Targets2:
    """round 2: paired section contrast (out/plan/targets2.json)"""

    def __init__(self, path=None):
        p = path or os.path.join(ROOT, "out", "plan", "targets2.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        self.d = d
        self.rel = d["rel"]
        self.lev = d["level"]
        self.std = d["stdref"]
        self.within = d["elasticity"]["within"]
        self.between = d["elasticity"]["between"]

    def mu(self, label, metric, level):
        """paired contrast label / verse of `metric` at `level` (1.0 for the verse itself or when the label has no data)"""
        if label == "verse":
            return 1.0
        xs, ys = [], []
        for b in BANDS:
            cell = (self.rel[b].get(label) or {}).get(metric)
            if cell is not None:
                xs.append(lv_center(b))
                ys.append(cell["mu"])
        if not xs:
            return 1.0
        return float(np.interp(float(level), xs, ys))

    def level_mean(self, metric, level, stat="mean"):
        xs, ys = [], []
        for b in BANDS:
            cell = self.lev[b].get(metric)
            if cell:
                xs.append(lv_center(b))
                ys.append(cell[stat])
        return float(np.interp(float(level), xs, ys)) if xs else float("nan")

    def verse_ref(self, metric, level, stat="p50"):
        xs, ys = [], []
        for b in BANDS:
            cell = (self.std[b]["V"].get(metric))
            if cell:
                xs.append(lv_center(b))
                ys.append(cell[stat])
        return float(np.interp(float(level), xs, ys)) if xs else float("nan")

    def slope(self, label, group):
        e = (self.within.get(group) or {}).get(label) or (self.within.get("all") or {}).get(label)
        return float(e["slope"]) if e else 0.0


_AB = {}


def fit_level_density(path=None):
    """official per-song mean npb vs level: returns (a, b) of log(npb) = a + b * lv, fitted on the chart list of official_by_label.json"""
    p = path or os.path.join(ROOT, "out", "plan", "official_by_label.json")
    if p in _AB:
        return _AB[p]
    d = json.load(open(p, encoding="utf-8"))["charts"]
    lv = np.array([c["lv"] for c in d.values()])
    nb = np.array([np.mean(c["npb_bar"]) for c in d.values()])
    y = np.log(np.maximum(nb, 0.2))
    A = np.stack([np.ones_like(lv), lv], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    _AB[p] = (float(coef[0]), float(coef[1]))
    return _AB[p]


def level_density(level, ab=None):
    a, b = ab or (0.0, 0.0)
    return float(np.exp(a + b * float(level)))


# ------------------------------------------------------------------------------------------------ planning
def bar_labels(rec, nb):
    lab = np.full(nb, -1, np.int64)
    st = [int(x) for x in rec["starts"]]
    lb = [int(x) for x in rec["labels"]]
    for i, s in enumerate(st):
        e = st[i + 1] if i + 1 < len(st) else nb
        lab[max(s, 0):min(e, nb)] = lb[i]
    return lab


def _window_label(ids):
    return LABELS[int(np.bincount(ids[ids >= 0], minlength=len(LABELS)).argmax())] if (ids >= 0).any() else "other"


_T2 = None
_T1 = None


def plan(rec, tab, level, targets=None, ab=None, hl_frac=0.15, hl_weight=0.3, log=None):
    """rec: section record (nb, starts, labels[, mbar, mkind]); tab: bar table [nb,1218] of the SAME bar grid; level: target level.
    -> dict(song=..., windows=[...], summary=...).  Round-2 planner unless a round-1 `Targets` object is passed."""
    global _T2
    if isinstance(targets, Targets):
        return plan_v1(rec, tab, level, targets=targets, ab=ab, hl_frac=hl_frac, hl_weight=hl_weight, log=log)
    tg = targets or _T2 or Targets2()
    _T2 = tg if isinstance(tg, Targets2) else _T2
    ab = ab if ab is not None else fit_level_density()
    nb = int(min(len(tab), int(rec.get("nb", len(tab)))))
    tab = np.asarray(tab)[:nb]
    lab = bar_labels(rec, nb)
    lv = float(level)
    band = band_of(lv)
    group = "mid" if band in ("12", "12+", "13", "13+") else "high"
    inten = PF.intensity(tab)
    lz = inten["loud_zs"]
    base_dens = level_density(lv, ab)
    nw = int(np.ceil(nb / WIN))
    weight = np.array([min((w + 1) * WIN, nb) - w * WIN for w in range(nw)], np.float64)
    names = []
    for w in range(nw):
        a, b = w * WIN, min((w + 1) * WIN, nb)
        names.append(_window_label(lab[a:b]))
    # ---- density: label contrast x between-label loudness x within-label loudness
    btw = float(tg.between[group]["slope"])
    typ_lz = tg.between[group]["mean_lz"]
    lbar = {}
    for k, name in enumerate(LABELS):
        sel = lab == k
        if sel.sum() >= 2:
            lbar[name] = float(lz[sel].mean())
    raw = np.zeros(nw)
    for w in range(nw):
        a, b = w * WIN, min((w + 1) * WIN, nb)
        name = names[w]
        dev_b = float(np.clip(lbar.get(name, typ_lz.get(name, 0.0)) - typ_lz.get(name, 0.0), -1.5, 1.5))
        dev_w = float(np.clip(np.mean(lz[a:b]) - lbar.get(name, np.mean(lz[a:b])), -2.0, 2.0))
        raw[w] = tg.mu(name, "npb", lv) * float(np.exp(btw * dev_b + tg.slope(name, group) * dev_w))
    cur = float((raw * weight).sum() / max(weight.sum(), 1))
    scale = base_dens / max(cur, 1e-9)
    npb_w = raw * scale
    notes_w = npb_w * weight
    # ---- shares / rates: paired contrast, song mean preserved
    mult = {}
    for m in MULT_KEYS:
        mu = np.array([tg.mu(n, m, lv) for n in names])
        z = float((mu * notes_w).sum() / max(notes_w.sum(), 1e-9))
        mult[m] = mu / max(z, 1e-9)
    absval = {}
    for m in MULT_KEYS:
        absval[m] = tg.level_mean(m, lv) * mult[m]
    # entropy / novelty / repetition metrics: verse level x paired contrast (no normalisation)
    for m in ("nov_lane", "motif_half", "repeat_prev", "slide_ent", "rhythm_ent", "lane_ent"):
        ref = tg.verse_ref(m, lv)
        absval[m] = np.array([ref * tg.mu(n, m, lv) for n in names])
    k_p90 = tg.level_mean("npb_p90", lv, "p50") / max(tg.level_mean("npb", lv, "p50"), 1e-6)
    absval["npb_p90"] = npb_w * k_p90
    # ---- highlight bars (loud, non chorus) -> chorus weight, same rule as mai/sections.window_weights
    sm = np.convolve(lz, np.ones(2) / 2, mode="same") if nb >= 4 else lz
    thr = float(np.quantile(sm, 1.0 - hl_frac)) if nb >= 8 else 1e9
    hl = (sm >= thr)
    nov_ref = tg.verse_ref("nov_lane", lv) * tg.mu("chorus", "nov_lane", lv)
    mot_v = tg.verse_ref("motif_half", lv)
    wins = []
    for w in range(nw):
        a, b = w * WIN, min((w + 1) * WIN, nb)
        ids = lab[a:b]
        name = names[w]
        ischo = np.isin(ids, CHORUS_IDS)
        wgt_c = float(np.mean(np.where(ischo, 1.0, np.where(hl[a:b], hl_weight, 0.0))))
        bv = np.zeros(len(ids))
        bv[ids == VERSE_ID] = 1.0
        bv[np.isin(ids, HALF_CALM)] = 0.5
        bv[ischo] = 0.0
        v = float(bv.mean()) if len(bv) else 0.0
        rel = float(npb_w[w] / max(base_dens, 1e-6))
        bucket = int(np.searchsorted(PLAN_EDGES, rel))
        win = dict(bar=int(a), bar_end=int(b), label=name, w=round(float(wgt_c), 3), v=round(float(v), 3),
                   loud_z=round(float(np.mean(lz[a:b])), 3),
                   npb=round(float(npb_w[w]), 3), notes=round(float(notes_w[w]), 1),
                   rel_density=round(rel, 3), plan_bucket=bucket, plan_delta=bucket - 4, density_mult=round(rel, 3))
        for k in RATE_KEYS:
            x = absval[k]
            win[k] = round(float(x[w]), 4)
        win["mult"] = {m: round(float(mult[m][w]), 4) for m in MULT_KEYS}
        nov_w, mot_w = float(absval["nov_lane"][w]), float(absval["motif_half"][w])
        nd = (nov_w / max(nov_ref, 1e-6)) * ((1.0 - mot_w) / max(1.0 - mot_v, 1e-6))
        nd = float(np.clip(nd, 0.2, 5.0))
        win["creativity"] = dict(novelty_demand=round(nd, 3),
                                 content_temp=round(float(np.clip(nd ** 0.5, 0.9, 1.15)), 3),
                                 anti_repeat=round(float(np.clip(nd ** -0.5, 0.75, 1.25)), 3),
                                 motif=round(float(np.clip(nd ** -1.0, 0.6, 1.5)), 3))
        wins.append(win)
    return _finish(rec, nb, wins, weight, lv, band, group, base_dens, nw, version=2)


def _finish(rec, nb, wins, weight, lv, band, group, base_dens, nw, version):
    ch = [w for w in wins if w["label"] in ("chorus", "drop")]
    ve = [w for w in wins if w["label"] == "verse"]
    dens = np.array([w["npb"] for w in wins])
    summ = dict(level=lv, band=band, group=group, nb=nb, n_windows=nw, base_npb=round(base_dens, 3),
                npb_mean=round(float(np.average(dens, weights=weight)), 3),
                npb_chorus=round(float(np.mean([w["npb"] for w in ch])), 3) if ch else None,
                npb_verse=round(float(np.mean([w["npb"] for w in ve])), 3) if ve else None,
                target_CV=npb_ratio(ve, ch), chorus_windows=len(ch), verse_windows=len(ve),
                chorus_boost=chorus_boost(ve, ch), labels_used=sorted({w["label"] for w in wins}), planner_version=version)
    return dict(song=dict(nb=nb, starts=[int(x) for x in rec["starts"]], labels=[int(x) for x in rec["labels"]],
                          labels_txt=[LABELS[int(x)] for x in rec["labels"]],
                          mbar=[int(x) for x in rec.get("mbar", [])], mkind=[int(x) for x in rec.get("mkind", [])]),
                windows=wins, summary=summ)


def plan_v1(rec, tab, level, targets=None, ab=None, hl_frac=0.15, hl_weight=0.3, log=None):
    """round-1 planner (independent per-label p50 + un-centred elasticity).  Reproduces the round-1 e25 report."""
    tg = targets or Targets()
    nb = int(min(len(tab), int(rec.get("nb", len(tab)))))
    tab = np.asarray(tab)[:nb]
    lab = bar_labels(rec, nb)
    lv = float(level)
    band = band_of(lv)
    group = "mid" if band in ("12", "12+", "13", "13+") else "high"
    inten = PF.intensity(tab)
    lz = inten["loud_zs"]
    table = {name: tg.label_table(name, lv, group) for name in LABELS}
    slope = {name: tg.elasticity(name, group) for name in LABELS}
    base_dens = level_density(lv, ab if ab is not None else fit_level_density())
    nw = int(np.ceil(nb / WIN))
    wins = []
    raw = np.zeros(nw)
    for w in range(nw):
        a, b = w * WIN, min((w + 1) * WIN, nb)
        name = _window_label(lab[a:b])
        lw = float(np.clip(np.mean(lz[a:b]), -2.0, 2.0))
        raw[w] = table[name]["npb"] * float(np.exp(slope[name] * lw))
    weight = np.array([min((w + 1) * WIN, nb) - w * WIN for w in range(nw)], np.float64)
    cur = float((raw * weight).sum() / max(weight.sum(), 1))
    scale = base_dens / max(cur, 1e-6)
    sm = np.convolve(lz, np.ones(2) / 2, mode="same") if nb >= 4 else lz
    thr = float(np.quantile(sm, 1.0 - hl_frac)) if nb >= 8 else 1e9
    hl = (sm >= thr)
    for w in range(nw):
        a, b = w * WIN, min((w + 1) * WIN, nb)
        ids = lab[a:b]
        name = _window_label(ids)
        ischo = np.isin(ids, CHORUS_IDS)
        wgt_c = float(np.mean(np.where(ischo, 1.0, np.where(hl[a:b], hl_weight, 0.0))))
        bv = np.zeros(len(ids))
        bv[ids == VERSE_ID] = 1.0
        bv[np.isin(ids, HALF_CALM)] = 0.5
        bv[ischo] = 0.0
        v = float(bv.mean()) if len(bv) else 0.0
        npb_t = float(raw[w] * scale)
        rel = float(npb_t / max(base_dens, 1e-6))
        bucket = int(np.searchsorted(PLAN_EDGES, rel))
        t = table[name]
        nov_ref = tg.value("chorus", "nov_lane", lv)
        nd = (t["nov_lane"] / max(nov_ref, 1e-6)) * ((1.0 - t["motif_half"]) / max(1.0 - tg.value("verse", "motif_half", lv), 1e-6))
        nd = float(np.clip(nd, 0.2, 5.0))
        win = dict(bar=int(a), bar_end=int(b), label=name, w=round(float(wgt_c), 3), v=round(float(v), 3),
                   loud_z=round(float(np.mean(lz[a:b])), 3), npb=round(npb_t, 3), notes=round(float(npb_t * (b - a)), 1),
                   rel_density=round(rel, 3), plan_bucket=bucket, plan_delta=bucket - 4, density_mult=round(rel, 3))
        for k in RATE_KEYS:
            win[k] = round(float(t[k]), 4)
        win["creativity"] = dict(novelty_demand=round(nd, 3),
                                 content_temp=round(float(np.clip(nd ** 0.5, 0.9, 1.15)), 3),
                                 anti_repeat=round(float(np.clip(nd ** -0.5, 0.75, 1.25)), 3),
                                 motif=round(float(np.clip(nd ** -1.0, 0.6, 1.5)), 3))
        wins.append(win)
    return _finish(rec, nb, wins, weight, lv, band, group, base_dens, nw, version=1)


def npb_ratio(ve, ch):
    if not ve or not ch:
        return None
    a = float(np.mean([w["npb"] for w in ch]))
    b = float(np.mean([w["npb"] for w in ve]))
    return round(a / max(b, 1e-6), 3)


def chorus_boost(ve, ch, cb_density=0.10):
    r = npb_ratio(ve, ch)
    if r is None:
        return 0.0
    return round(float(np.clip((r - 1.0) / cb_density / 2.0, 0.0, 3.0)), 2)


# ------------------------------------------------------------------------------------------------ consumer helpers
def load_plan(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def window_arrays(plan):
    """planner JSON -> dict of per-window numpy arrays, indexed by the 4-bar window number (bar // 4), i.e. the same index the
    generator uses for `expect.expected` and for `Steer.begin_window`.  Missing windows stay NaN.  `mult_<metric>` arrays hold the
    relative multipliers (round 2; absent for a round-1 plan)."""
    W = plan["windows"]
    n = (max(w["bar_end"] for w in W) + WIN - 1) // WIN if W else 0
    keys = ["npb", "notes", "rel_density", "plan_bucket", "plan_delta", "w", "v", "loud_z"] + list(RATE_KEYS)
    out = {k: np.full(n, np.nan) for k in keys}
    out["label"] = np.array([""] * n, dtype=object)
    for w in W:
        i = int(w["bar"]) // WIN
        if 0 <= i < n:
            for k in keys:
                out[k][i] = float(w[k])
            out["label"][i] = w["label"]
    for c in ("content_temp", "anti_repeat", "motif"):
        out[c] = np.full(n, np.nan)
        for w in W:
            i = int(w["bar"]) // WIN
            if 0 <= i < n:
                out[c][i] = float(w["creativity"][c])
    if W and "mult" in W[0]:
        for m in W[0]["mult"]:
            arr = np.full(n, np.nan)
            for w in W:
                i = int(w["bar"]) // WIN
                if 0 <= i < n:
                    arr[i] = float(w["mult"][m])
            out["mult_" + m] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int)
    ap.add_argument("--slot", type=int, default=5)
    ap.add_argument("--level", type=float)
    ap.add_argument("--sections", default="", help="section JSON (v4 record); empty = run sectionpred on the bar table")
    ap.add_argument("--bars", default=os.environ.get("MAI_BARS_ROOT") or r"E:\maiAIV3_feats\bars_v3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import sectionpred as SP
    tab = np.asarray(np.load(os.path.join(a.bars, f"{a.sid:05d}_{a.slot}.npy"), mmap_mode="r"), np.float32)
    if a.sections:
        rec = json.load(open(a.sections, encoding="utf-8"))
        rec = rec.get("song", rec)
    else:
        rec = SP.predict_from_table(tab)
    res = plan(rec, tab, a.level, ab=fit_level_density())
    s = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(s)
        print("wrote", a.out, flush=True)
    else:
        print(s)


if __name__ == "__main__":
    main()

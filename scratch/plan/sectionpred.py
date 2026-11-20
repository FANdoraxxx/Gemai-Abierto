"""Audio-only song-structure predictor: per-bar section label + section boundaries + highlight moments.

Input  = the song's *bar table* (mai/v3.compute_bar_table: MERT L8 + mel statistics per 4-beat bar), i.e. AUDIO + BAR GRID only.
Output = a v4 annotation record ``dict(nb, starts, labels, mbar, mkind)`` -- exactly what ``scratch/v4/mai/ann.py`` builds from the
         Gemini answers and what ``mai/sections.py`` / ``--chorus-sections`` consumes, so 'predicted' and 'Gemini' sections are
         interchangeable downstream.

Model   = per-bar gradient boosting (HistGradientBoostingClassifier) on the planfeat.py features (86 dims: MERT structure PCA,
          self-similarity / recurrence, loudness / spectrum / flux, song position), then Viterbi decoding over the 10-label
          vocabulary with the label-transition matrix counted on the training songs, a stickiness temperature and a minimum section
          length.  Three extra boosters score the moments (big_hit / riser / drop).

Commands
  .venv\\Scripts\\python.exe scratch\\plan\\sectionpred.py --mode cv          # 5-fold song-level CV (folds = hash(sid) % 5)
  .venv\\Scripts\\python.exe scratch\\plan\\sectionpred.py --mode final       # train on all train songs -> out/plan/sectionpred.pkl
  .venv\\Scripts\\python.exe scratch\\plan\\sectionpred.py --mode predict --sid 41 --slot 5

Training songs = the 973 songs of work/charts.pkl that have a Gemini annotation *and* a bar table and are not in work/val_songs.txt
(which contains all of e25).  Nothing of val_songs / e25 is ever fitted on.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scratch", "plan"))
import planfeat as PF  # noqa: E402

N_SEC = 10
N_MOM = 3
LABELS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "break", "build-up", "drop", "outro", "other")
MOMENT_KINDS = ("big_hit", "riser", "drop")
CHORUS_IDS = (3, 7)
MODEL_PATH = os.path.join(ROOT, "out", "plan", "sectionpred.pkl")


# --------------------------------------------------------------------------------------------- training -------------------------------------------------------
def _gbm(iters, seed=0, leaves=63):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.1, max_leaf_nodes=leaves, min_samples_leaf=30,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)


def fit_model(songs, iters=200, mom_iters=80, seed=0, log=print):
    """songs: list of dicts with F / lab / mom -> model dict"""
    t0 = time.time()
    X = np.concatenate([s["F"] for s in songs])
    y = np.concatenate([s["lab"] for s in songs])
    lab = _gbm(iters, seed).fit(X, y)
    log(f"  label GBM fitted ({len(y)} bars, {time.time() - t0:.0f}s)", flush=True)
    moms = []
    Y = np.concatenate([s["mom"] for s in songs])
    for m in range(N_MOM):
        yy = (Y[:, m] > 0.5).astype(int)
        if yy.sum() < 10:
            moms.append(None)
            continue
        moms.append(_gbm(mom_iters, seed + m, leaves=31).fit(X, yy))
    log(f"  moment GBMs fitted ({time.time() - t0:.0f}s)", flush=True)
    import train as T
    logT = T.transition_matrix(songs)
    logDur = T.length_prior(songs)
    return dict(kind="gbm", label=lab, moms=moms, logT=logT, logDur=logDur, dur_scale=1.0, stick=1.0, min_len=3,
                labels=LABELS, n_train=len(songs))


def song_preds(model, songs, bs=None):
    """per-song dict(logp, bnd, mom) for the decoder"""
    X = np.concatenate([s["F"] for s in songs])
    proba = model["label"].predict_proba(X)
    cls = np.asarray(model["label"].classes_, int)
    P = np.zeros((len(X), N_SEC), np.float32)
    P[:, cls] = proba
    P = np.maximum(P, 1e-6)
    P /= P.sum(1, keepdims=True)
    mom = np.zeros((len(X), N_MOM), np.float32)
    for m, g in enumerate(model["moms"]):
        if g is not None:
            mom[:, m] = g.predict_proba(X)[:, 1]
    out, o = [], 0
    for s in songs:
        k = len(s["F"])
        p = P[o:o + k]
        bnd = np.zeros(k, np.float32)
        if k > 1:
            bnd[1:] = 1.0 - np.sum(p[1:] * p[:-1], 1)
        out.append(dict(logp=np.log(p), bnd=bnd, mom=mom[o:o + k]))
        o += k
    return out


def decode(model, pred, min_len=None, dur_scale=None, bnd_boost=0.0):
    import train as T
    return T.viterbi_dur(pred["logp"], model["logT"], min_len=model["min_len"] if min_len is None else min_len,
                         max_len=64, logDur=model["logDur"],
                         dur_scale=model["dur_scale"] if dur_scale is None else dur_scale,
                         bnd=pred["bnd"], bnd_boost=bnd_boost)


def record_from_path(path, mom, thr=0.5):
    st, lab = _runs(path)
    mbar, mkind = [], []
    for m in range(N_MOM):
        idx = np.flatnonzero(mom[:, m] >= thr)
        # keep the local maxima only (one moment per run)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1):
            if len(run) == 0:
                continue
            b = int(run[np.argmax(mom[run, m])])
            if (b, m) not in zip(mbar, mkind):
                mbar.append(b)
                mkind.append(m)
    order = np.argsort(mbar) if mbar else []
    return dict(nb=int(len(path)), starts=[int(x) for x in st], labels=[int(x) for x in lab],
                mbar=[int(mbar[i]) for i in order], mkind=[int(mkind[i]) for i in order])


def _runs(path):
    st = np.flatnonzero(np.r_[True, path[1:] != path[:-1]])
    return st, path[st]


# --------------------------------------------------------------------------------------------- inference API ---------------------------------------------------
_MODEL = None


def load(path=None):
    global _MODEL
    p = path or MODEL_PATH
    with open(p, "rb") as f:
        _MODEL = pickle.load(f)
    return _MODEL


def section_records(tab, nb=None, model=None, stick=None, min_len=None):
    """bar table [nb,1218] -> list of candidate records for the given stick/min_len (fast path used by the planner)"""
    m = model or _MODEL or load()
    nb = int(min(len(tab), nb) if nb else len(tab))
    F = PF.clip_feats(PF.bar_feats(np.asarray(tab)[:nb], m["pca_mean"], m["pca_comp"]))
    preds = song_preds(m, [dict(F=F)])
    out = {}
    for st in (stick if stick is not None else [m.get("dur_scale", 1.0)]):
        for ml in (min_len if min_len is not None else [m["min_len"]]):
            out[(st, ml)] = record_from_path(decode(m, preds[0], ml, st), preds[0]["mom"])
    return out


def predict_from_table(tab, nb=None):
    """drop-in for mai/sections.get_record: -> record dict (nb, starts, labels, mbar, mkind)"""
    m = _MODEL or load()
    st = _MODEL_STICK if _MODEL_STICK is not None else m.get("dur_scale", 1.0)
    ml = _MODEL_MINLEN if _MODEL_MINLEN is not None else m["min_len"]
    return section_records(tab, nb, model=m, stick=[st], min_len=[ml])[(st, ml)]


_MODEL_STICK, _MODEL_MINLEN = None, None       # None = use the values stored in the model


def set_defaults(stick, min_len):
    global _MODEL_STICK, _MODEL_MINLEN
    _MODEL_STICK, _MODEL_MINLEN = float(stick), int(min_len)


# --------------------------------------------------------------------------------------------- CV / final ------------------------------------------------------
def _cv_worker(fold, ds, iters, mom_iters):
    import train as T
    songs = _load(ds)[0]
    tr = [s for s in songs if s["split"] == "train" and s["fold"] != fold]
    va = [s for s in songs if s["split"] == "train" and s["fold"] == fold]
    m = fit_model(tr, iters=iters, mom_iters=mom_iters, log=lambda *a, **k: None)
    m.update(pca_mean=_PCA[0], pca_comp=_PCA[1])
    preds = song_preds(m, va)
    grid = {}
    for dur_scale in (0.0, 0.5, 1.0):
        for ml in (1, 2, 3, 4, 6):
            ev = T.eval_songs(va, preds, m["logT"], min_len=ml, dec="dur", logDur=m["logDur"], dur_scale=dur_scale)
            grid[(dur_scale, ml)] = {k: v for k, v in ev.items() if k != "conf"}
    best = max(grid.items(), key=lambda kv: kv[1]["acc"] + 0.5 * kv[1]["bnd_f1"])
    maj = int(np.bincount(np.concatenate([s["lab"] for s in tr]), minlength=N_SEC).argmax())
    base = float(np.mean(np.concatenate([(s["lab"] == maj).astype(float) for s in va])))
    return fold, dict(grid={f"{k[0]},{k[1]}": v for k, v in grid.items()}, best_key=list(best[0]), best=best[1],
                      majority=dict(label=maj, acc=base))


_PCA = (None, None)


def _cv_star(t):
    return _cv_worker(*t)


def _load(ds):
    import train as T
    songs, pca = T.load_ds(ds)
    global _PCA
    _PCA = (pca["pca_mean"], pca["pca_comp"])
    return songs, pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cv", choices=("cv", "final", "predict", "grid"))
    ap.add_argument("--ds", default=os.path.join(ROOT, "out", "plan", "ds.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "plan"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--mom-iters", type=int, default=80, dest="mom_iters")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--sid", type=int)
    ap.add_argument("--slot", type=int, default=5)
    ap.add_argument("--stick", type=float, default=None)
    ap.add_argument("--min-len", type=int, default=None, dest="min_len")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode in ("cv", "grid"):
        import multiprocessing as mp
        folds = [int(x) for x in a.folds.split(",")]
        t0 = time.time()
        res = {}
        if a.procs > 1:
            with mp.get_context("spawn").Pool(len(folds)) as pool:
                for fold, r in pool.imap_unordered(_cv_star, [(f, a.ds, a.iters, a.mom_iters) for f in folds]):
                    res[fold] = r
                    print(f"fold {fold}: acc {r['best']['acc']:.4f} bndF1 {r['best']['bnd_f1']:.3f} "
                          f"chorusF1 {r['best']['chorus_f1']:.3f} secRecog {r['best']['sec_recog']:.3f} "
                          f"stick {r['best_key'][0]} minlen {r['best_key'][1]} [{time.time() - t0:.0f}s]", flush=True)
        else:
            for f in folds:
                fold, r = _cv_worker(f, a.ds, a.iters, a.mom_iters)
                res[fold] = r
                print(f"fold {fold}: {r['best']}", flush=True)
        keys = ("acc", "macro_f1", "bnd_f1", "bnd_p", "bnd_r", "chorus_f1", "chorus_p", "chorus_r", "sec_recog")
        summ = {k: dict(mean=float(np.mean([res[f]["best"][k] for f in res])), std=float(np.std([res[f]["best"][k] for f in res])))
                for k in keys}
        summ["majority_acc"] = float(np.mean([res[f]["majority"]["acc"] for f in res]))
        summ["mom_f1"] = [float(np.mean([res[f]["best"]["mom_f1"][m] for f in res])) for m in range(N_MOM)]
        # grid: pooled mean over folds
        grid = {}
        for key in res[list(res)[0]]["grid"]:
            grid[key] = {k: float(np.mean([res[f]["grid"][key][k] for f in res]))
                         for k in ("acc", "macro_f1", "bnd_f1", "chorus_f1", "sec_recog")}
        print("CV summary:", json.dumps(summ, indent=1), flush=True)
        json.dump(dict(summary=summ, folds=res, grid=grid, args=vars(a)), open(os.path.join(a.out, "cv_gbm.json"), "w"), indent=1)
    elif a.mode == "final":
        songs, pca = _load(a.ds)
        tr = [s for s in songs if s["split"] == "train"]
        va = [s for s in songs if s["split"] == "val"]
        t0 = time.time()
        m = fit_model(tr, iters=a.iters, mom_iters=a.mom_iters)
        m.update(pca_mean=pca["pca_mean"], pca_comp=pca["pca_comp"])
        import train as T2
        # choose stick / min_len: best grid on the val songs is forbidden -> use the CV-chosen constants passed in
        st = a.stick if a.stick is not None else 1.0
        ml = a.min_len if a.min_len is not None else 3
        m["dur_scale"], m["min_len"] = st, ml
        with open(os.path.join(a.out, "sectionpred.pkl"), "wb") as f:
            pickle.dump(m, f)
        ev = T2.eval_songs(va, song_preds(m, va), m["logT"], min_len=ml, dec="dur", logDur=m["logDur"], dur_scale=st)
        json.dump(dict(val=ev, n_train=len(tr), stick=st, min_len=ml), open(os.path.join(a.out, "final_gbm.json"), "w"), indent=1)
        print(f"saved {os.path.join(a.out, 'sectionpred.pkl')} ({time.time() - t0:.0f}s); val acc {ev['acc']:.4f} "
              f"bndF1 {ev['bnd_f1']:.3f} chorusF1 {ev['chorus_f1']:.3f}", flush=True)
    elif a.mode == "predict":
        m = load()
        tab = np.load(os.path.join(os.environ.get("MAI_BARS_ROOT") or r"E:\maiAIV3_feats\bars_v3", f"{a.sid:05d}_{a.slot}.npy"),
                      mmap_mode="r")
        rec = section_records(np.asarray(tab, np.float32), model=m)[(m["dur_scale"], m["min_len"])]
        rec["labels_txt"] = [LABELS[int(x)] for x in rec["labels"]]
        print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()

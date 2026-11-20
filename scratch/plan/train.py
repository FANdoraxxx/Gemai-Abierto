"""Train / cross-validate the audio-only section predictor (CPU, torch) -> out/plan/model.pt + out/plan/cv.json.

  .venv\\Scripts\\python.exe scratch\\plan\\train.py --mode cv          # song-level 5-fold CV, report metrics
  .venv\\Scripts\\python.exe scratch\\plan\\train.py --mode final       # train on all train songs, save model

Targets per bar: 10 section labels (v4/Gemini vocabulary), a section-start flag, and the 3 moment flags (big_hit / riser / drop).
Post-processing = Viterbi over the label posteriors with the label-transition matrix counted on the training songs (plus a stickiness
temperature and a minimum section length), which is what removes the isolated per-bar flicker of a framewise classifier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scratch", "plan"))
import planfeat as PF  # noqa: E402

N_SEC = 10
N_MOM = 3
LABELS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "break", "build-up", "drop", "outro", "other")
CHORUS_IDS = (3, 7)


# --------------------------------------------------------------------------------------------- dataset
def load_ds(path):
    z = np.load(path, allow_pickle=False)
    F = z["F"].astype(np.float32)
    lab = z["lab"].astype(np.int64)
    off = z["m_off"]
    mb, mk = z["mbar"], z["mkind"]
    songs = []
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        nb = b - a
        starts = np.flatnonzero(np.r_[True, lab[a + 1:b] != lab[a:b - 1]]) if nb > 1 else np.zeros(1, np.int64)
        mom = np.zeros((nb, N_MOM), np.float32)
        for bar, k in zip(mb, mk):
            if a <= bar < b and 0 <= k < N_MOM:
                mom[int(bar) - a, int(k)] = 1.0
        bnd = np.zeros(nb, np.float32)
        inner = np.asarray(starts[1:], np.int64)
        bnd[inner[(inner >= 0) & (inner < nb)]] = 1.0
        songs.append(dict(idx=i, sid=int(z["m_sid"][i]), folder=str(z["m_folder"][i]), F=F[a:b], lab=lab[a:b],
                          bnd=bnd, starts=starts,
                          mom=mom, fold=int(z["m_fold"][i]), split=str(z["m_split"][i])))
    return songs, dict(pca_mean=z["pca_mean"], pca_comp=z["pca_comp"])


# --------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d_in, h=128, layers=2, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(drop))
        self.rnn = nn.GRU(h, h, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=drop if layers > 1 else 0.0)
        self.out_lab = nn.Linear(2 * h, N_SEC)
        self.out_bnd = nn.Linear(2 * h, 1)
        self.out_mom = nn.Linear(2 * h, N_MOM)
        self.drop = nn.Dropout(drop)

    def forward(self, x, lengths):
        h = self.inp(x)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        y = self.drop(y)
        return self.out_lab(y), self.out_bnd(y).squeeze(-1), self.out_mom(y)


def collate(batch):
    n = max(len(s["F"]) for s in batch)
    B, d = len(batch), batch[0]["F"].shape[1]
    X = np.zeros((B, n, d), np.float32)
    Y = np.zeros((B, n), np.int64)
    Bd = np.zeros((B, n), np.float32)
    Mo = np.zeros((B, n, N_MOM), np.float32)
    M = np.zeros((B, n), np.float32)
    for i, s in enumerate(batch):
        k = len(s["F"])
        X[i, :k] = s["F"]
        Y[i, :k] = s["lab"]
        Bd[i, :k] = s["bnd"]
        Mo[i, :k] = s["mom"]
        M[i, :k] = 1.0
    return (torch.from_numpy(X), torch.tensor([len(s["F"]) for s in batch]), torch.from_numpy(Y),
            torch.from_numpy(Bd), torch.from_numpy(Mo), torch.from_numpy(M))


# --------------------------------------------------------------------------------------------- post-processing
def transition_matrix(songs, alpha=1.0):
    """label bigram counts of the training songs -> log transition matrix (smoothed), 10x10"""
    C = np.full((N_SEC, N_SEC), alpha)
    for s in songs:
        tr = s["lab"][1:][s["lab"][1:] != s["lab"][:-1]]
        if len(tr) == 0:
            continue
        prev = s["lab"][np.flatnonzero(s["lab"][1:] != s["lab"][:-1])]
        for a, b in zip(prev, tr):
            C[a, b] += 1.0
    P = C / C.sum(1, keepdims=True)
    return np.log(P)


def viterbi(logp, logT, stick=1.0, bnd_boost=0.0, bnd=None):
    """logp [nb,10] emission log-probs -> label path.  stick scales the diagonal (higher = longer sections);
    bnd_boost adds a bonus to any label change at bars with a high predicted boundary score"""
    nb = len(logp)
    T = logT.copy()
    np.fill_diagonal(T, np.diag(logT) + np.log(max(stick, 1e-6)))
    dp = np.full((nb, N_SEC), -1e18)
    bk = np.zeros((nb, N_SEC), np.int16)
    dp[0] = logp[0]
    off = np.eye(N_SEC, dtype=bool)
    Tt = T.T                                   # [prev, cur]
    for b in range(1, nb):
        cand = dp[b - 1][:, None] + Tt         # [prev, cur]
        if bnd is not None and bnd_boost and bnd[b] > 0.5:
            cand = cand + bnd_boost * ~off
        bk[b] = cand.argmax(0)
        dp[b] = cand.max(0) + logp[b]
    path = np.zeros(nb, np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for b in range(nb - 1, 0, -1):
        path[b - 1] = bk[b, path[b]]
    return path


def merge_short(path, min_len):
    """merge runs shorter than min_len bars into the neighbouring run with the largest label similarity is not known here ->
    merge into the left run (or the right one for the first run)"""
    if min_len <= 1 or len(path) == 0:
        return path
    p = path.copy()
    changed = True
    while changed:
        changed = False
        starts = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
        ends = np.r_[starts[1:], len(p)]
        for i, (a, e) in enumerate(zip(starts, ends)):
            if e - a >= min_len or len(starts) == 1:
                continue
            if i == 0:
                p[a:e] = p[e] if e < len(p) else p[a]
            else:
                p[a:e] = p[a - 1]
            changed = True
            break
    return p


def length_prior(songs, max_len=64, alpha=1.0):
    """log duration prior of a section (in bars), counted on the training songs: log((n_L + alpha) / sum)"""
    h = np.full(max_len + 1, alpha)
    for s in songs:
        st = np.r_[s["starts"], len(s["lab"])]
        for a, b in zip(st[:-1], st[1:]):
            L = int(min(b - a, max_len))
            if L >= 1:
                h[L] += 1.0
    return np.log(h / h.sum())


def viterbi_dur(logp, logT, min_len=2, max_len=64, logDur=None, stick=1.0, dur_scale=1.0, bnd=None, bnd_boost=0.0):
    """semi-Markov Viterbi: every section lasts min_len..max_len bars, transition matrix logT, optional duration prior logDur.
    Returns the label path [nb].  (Framewise Viterbi + run merging cascades; this one enforces the minimum length exactly.)"""
    nb = len(logp)
    if nb == 0:
        return np.zeros(0, np.int64)
    T = logT.copy()
    np.fill_diagonal(T, -1e9)          # consecutive segments must switch label; stickiness comes from the duration prior
    cum = np.concatenate([np.zeros((1, N_SEC)), np.cumsum(logp, 0)], 0)     # [nb+1, 10]
    if logDur is None:
        logDur = np.zeros(max_len + 1)
    logDur = np.asarray(logDur, np.float64)[: max_len + 1] * float(dur_scale)
    dp = np.full((nb + 1, N_SEC), -1e18)
    dp[0] = 0.0
    bkL = np.zeros((nb + 1, N_SEC), np.int32)
    bkI = np.zeros((nb + 1, N_SEC), np.int32)
    ml = max(1, int(min_len))
    for t in range(ml, nb + 1):
        Ls = np.arange(ml, min(t, max_len) + 1)
        if len(Ls) == 0:
            continue
        S = cum[t][None, :] - cum[t - Ls]                       # [nL, 10]
        A = dp[t - Ls]                                          # [nL, 10]  (rows for unreachable starts are -1e18)
        first = Ls == t
        base = S + logDur[Ls][:, None]
        trans = np.where(first[:, None, None], 0.0, A[:, :, None] + T[None, :, :])   # [nL, prev, cur]
        if bnd is not None and bnd_boost and t >= 1 and bnd[t - 1] > 0.5:
            trans = trans + bnd_boost
        sc = trans + base[:, None, :]                                            # [nL, prev, cur]
        sc = sc.reshape(len(Ls) * N_SEC, N_SEC)
        idx = sc.argmax(0)
        dp[t] = sc[idx, np.arange(N_SEC)]
        bkL[t] = Ls[idx // N_SEC]
        bkI[t] = idx % N_SEC
    # backtrack
    path = np.zeros(nb, np.int64)
    t, j = nb, int(dp[nb].argmax())
    while t > 0:
        L = int(bkL[t, j])
        path[t - L:t] = j
        if t - L > 0:
            j = int(bkI[t, j])
        t -= L
    return path


def runs(path):
    """label path -> (starts, labels)"""
    st = np.flatnonzero(np.r_[True, path[1:] != path[:-1]])
    return st, path[st]


# --------------------------------------------------------------------------------------------- train / eval
def predict_probs(model, songs, bs=8):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(songs), bs):
            batch = songs[i:i + bs]
            X, L, *_ = collate(batch)
            pl, pb, pm = model(X, L)
            for j, s in enumerate(batch):
                k = len(s["F"])
                out.append(dict(logp=Fn.log_softmax(pl[j, :k], -1).numpy(),
                                bnd=torch.sigmoid(pb[j, :k]).numpy(),
                                mom=torch.sigmoid(pm[j, :k]).numpy()))
    return out


def eval_songs(songs, preds, logT, stick=1.0, min_len=1, bnd_boost=0.0, dec="frame", logDur=None, dur_scale=1.0):
    """-> overall metrics dict (per-bar acc, macro F1, boundary F1 +-1 bar, chorus P/R/F1, section recognition)"""
    nb_tot = 0
    acc = 0
    conf = np.zeros((N_SEC, N_SEC), np.int64)
    b_tp = b_fp = b_fn = 0
    c_tp = c_fp = c_fn = 0
    rec_ok = rec_n = 0
    mom_tp = np.zeros(N_MOM)
    mom_fp = np.zeros(N_MOM)
    mom_fn = np.zeros(N_MOM)
    for s, p in zip(songs, preds):
        if dec == "dur":
            path = viterbi_dur(p["logp"], logT, min_len=min_len, max_len=64, logDur=logDur, dur_scale=dur_scale,
                               bnd=p["bnd"], bnd_boost=bnd_boost)
        else:
            path = merge_short(viterbi(p["logp"], logT, stick, bnd_boost, p["bnd"]), min_len)
        ref = s["lab"]
        nb_tot += len(ref)
        acc += int((path == ref).sum())
        for a, b in zip(ref, path):
            conf[a, b] += 1
        # boundaries +-1 bar, one-to-one greedy
        ps = np.flatnonzero(np.r_[True, path[1:] != path[:-1]])[1:]
        rs = s["starts"][1:]
        used = np.zeros(len(ps), bool)
        for r in rs:
            cand = np.flatnonzero((~used) & (np.abs(ps - r) <= 1))
            if len(cand):
                j = cand[np.argmin(np.abs(ps[cand] - r))]
                used[j] = True
                b_tp += 1
            else:
                b_fn += 1
        b_fp += int((~used).sum())
        # chorus / drop bars
        rc = np.isin(ref, CHORUS_IDS)
        pc = np.isin(path, CHORUS_IDS)
        c_tp += int((rc & pc).sum())
        c_fp += int((pc & ~rc).sum())
        c_fn += int((rc & ~pc).sum())
        # section-level recognition of reference chorus / drop sections
        st = np.r_[s["starts"], len(ref)]
        for i, (a, b) in enumerate(zip(st[:-1], st[1:])):
            if ref[a] in CHORUS_IDS and b - a >= 4:
                rec_n += 1
                ov = (pc[a:b].sum()) / (b - a)
                lab_p = np.bincount(path[a:b], minlength=N_SEC)
                if ov >= 0.5 and np.isin(int(lab_p.argmax()), CHORUS_IDS):
                    rec_ok += 1
        for m in range(N_MOM):
            r = s["mom"][:, m] > 0.5
            q = p["mom"][:, m] > 0.5
            mom_tp[m] += (r & q).sum()
            mom_fp[m] += (~r & q).sum()
            mom_fn[m] += (r & ~q).sum()
    prec = c_tp / max(c_tp + c_fp, 1)
    rec = c_tp / max(c_tp + c_fn, 1)
    bp = b_tp / max(b_tp + b_fp, 1)
    br = b_tp / max(b_tp + b_fn, 1)
    f1s = []
    for k in range(N_SEC):
        tp = conf[k, k]
        pr = tp / max(conf[:, k].sum(), 1)
        rc_ = tp / max(conf[k, :].sum(), 1)
        f1s.append(2 * pr * rc_ / max(pr + rc_, 1e-9))
    f1m = [2 * mom_tp[m] / max(2 * mom_tp[m] + mom_fp[m] + mom_fn[m], 1) for m in range(N_MOM)]
    return dict(bars=nb_tot, acc=acc / max(nb_tot, 1), macro_f1=float(np.mean(f1s)), per_label_f1=[round(x, 4) for x in f1s],
                bnd_p=float(bp), bnd_r=float(br), bnd_f1=float(2 * bp * br / max(bp + br, 1e-9)), bnd_tp=b_tp, bnd_fp=b_fp, bnd_fn=b_fn,
                chorus_p=float(prec), chorus_r=float(rec), chorus_f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                sec_recog=float(rec_ok / max(rec_n, 1)), n_sec=rec_n,
                mom_f1=[round(float(x), 4) for x in f1m],
                conf=conf.tolist())


def train_one(tr, va, d_in, epochs=60, bs=8, lr=3e-3, wd=1e-2, seed=0, args=None, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(d_in, h=args.hidden, layers=args.layers, drop=args.dropout)
    cnt = np.bincount(np.concatenate([s["lab"] for s in tr]), minlength=N_SEC).astype(np.float64)
    wgt = torch.tensor((1.0 / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32)
    wgt = wgt * (N_SEC / wgt.sum())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(tr) // bs), pct_start=0.15)
    # per-feature scales for the noise augmentation
    allf = np.concatenate([s["F"] for s in tr], 0)
    sd = allf.std(0) + 1e-6
    best = (-1e9, None, None)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot, n = 0.0, 0
        for i in range(0, len(tr) - bs + 1, bs):
            batch = []
            for j in perm[i:i + bs]:
                s = tr[j]
                Fm = s["F"].copy()
                if args.noise > 0:
                    Fm += rng.normal(0, args.noise * sd, Fm.shape).astype(np.float32)
                if args.fdrop > 0:
                    k = rng.random(Fm.shape[1]) < args.fdrop
                    Fm[:, k] = 0.0
                batch.append(dict(s, F=Fm))
            X, L, Y, Bd, Mo, M = collate(batch)
            pl, pb, pm = model(X, L)
            T = Y.shape[1]
            m = M.bool()
            loss = (Fn.cross_entropy(pl[m], Y[m], weight=wgt, label_smoothing=0.05)
                    + args.w_bnd * Fn.binary_cross_entropy_with_logits(pb[m], Bd[m], pos_weight=torch.tensor(3.0))
                    + args.w_mom * Fn.binary_cross_entropy_with_logits(pm[m], Mo[m]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss) * len(batch)
            n += len(batch)
        if va is not None and (ep % 5 == 4 or ep == epochs - 1):
            preds = predict_probs(model, va)
            logT = transition_matrix(tr)
            ev = eval_songs(va, preds, logT, stick=1.0, min_len=2)
            hist.append(dict(ep=ep, loss=tot / max(n, 1), **{k: v for k, v in ev.items() if k != "conf"}))
            log(f"  ep {ep + 1:3d} train {tot / max(n, 1):.4f} val acc {ev['acc']:.4f} macroF1 {ev['macro_f1']:.4f} "
                f"bndF1 {ev['bnd_f1']:.3f} chorusF1 {ev['chorus_f1']:.3f}", flush=True)
            sc = ev["acc"] + 0.5 * ev["bnd_f1"]
            if sc > best[0]:
                best = (sc, {k: v.detach().clone() for k, v in model.state_dict().items()}, ev)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, hist, best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cv", choices=("cv", "final", "tune"))
    ap.add_argument("--ds", default=os.path.join(ROOT, "out", "plan", "ds.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "plan"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--fdrop", type=float, default=0.03)
    ap.add_argument("--w-bnd", type=float, default=0.5, dest="w_bnd")
    ap.add_argument("--w-mom", type=float, default=0.2, dest="w_mom")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    songs, pca = load_ds(a.ds)
    tr_all = [s for s in songs if s["split"] == "train"]
    va_all = [s for s in songs if s["split"] == "val"]
    d_in = tr_all[0]["F"].shape[1]
    print(f"{len(tr_all)} train songs, {len(va_all)} val songs, {d_in} features", flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "cv":
        folds = [int(x) for x in a.folds.split(",")]
        res, t0 = [], time.time()
        for f in folds:
            tr = [s for s in tr_all if s["fold"] != f]
            va = [s for s in tr_all if s["fold"] == f]
            print(f"fold {f}: {len(tr)} train / {len(va)} val", flush=True)
            model, hist, best = train_one(tr, va, d_in, epochs=a.epochs, bs=a.bs, lr=a.lr, seed=a.seed + f, args=a)
            logT = transition_matrix(tr)
            preds = predict_probs(model, va)
            ev = eval_songs(va, preds, logT, stick=1.0, min_len=2)
            # majority baseline on the same songs
            maj = int(np.bincount(np.concatenate([s["lab"] for s in tr]), minlength=N_SEC).argmax())
            base_acc = float(np.mean(np.concatenate([(s["lab"] == maj).astype(float) for s in va])))
            ev["majority"] = dict(label=maj, acc=base_acc)
            ev["fold"], ev["hist"] = f, hist
            res.append(ev)
            print(f"fold {f}: acc {ev['acc']:.4f} (majority {base_acc:.4f}) macroF1 {ev['macro_f1']:.4f} "
                  f"bndF1 {ev['bnd_f1']:.3f} chorus F1 {ev['chorus_f1']:.3f} P {ev['chorus_p']:.3f} R {ev['chorus_r']:.3f} "
                  f"secRecog {ev['sec_recog']:.3f} mom {ev['mom_f1']} [{time.time() - t0:.0f}s]", flush=True)
        keys = ("acc", "macro_f1", "bnd_f1", "chorus_f1", "chorus_p", "chorus_r", "sec_recog")
        summ = {k: dict(mean=float(np.mean([r[k] for r in res])), std=float(np.std([r[k] for r in res]))) for k in keys}
        summ["majority_acc"] = float(np.mean([r["majority"]["acc"] for r in res]))
        print("CV summary:", json.dumps(summ, indent=1), flush=True)
        json.dump(dict(summary=summ, folds=res, args=vars(a)), open(os.path.join(a.out, "cv.json"), "w"), indent=1)
    else:
        model, hist, best = train_one(tr_all, va_all, d_in, epochs=a.epochs, bs=a.bs, lr=a.lr, seed=a.seed, args=a)
        logT = transition_matrix(tr_all)
        torch.save(dict(state=model.state_dict(), d_in=d_in, hidden=a.hidden, layers=a.layers, dropout=a.dropout,
                        logT=logT, stick=1.0, min_len=2, labels=LABELS), os.path.join(a.out, "model.pt"))
        ev = eval_songs(va_all, predict_probs(model, va_all), logT, 1.0, 2)
        json.dump(dict(val=ev, args=vars(a), hist=hist), open(os.path.join(a.out, "final.json"), "w"), indent=1)
        print("saved model; val acc %.4f bndF1 %.3f chorus F1 %.3f" % (ev["acc"], ev["bnd_f1"], ev["chorus_f1"]), flush=True)


if __name__ == "__main__":
    main()

"""Frozen pretrained audio-encoder features (MERT / MuScriptor) -> work/feats/<encoder>/L<layer>/<sid:05d>.npy

Output per song/layer: float16 [T, D]; frame i is centred at time i / fps (fps 75 for MERT, 100 for MuScriptor).
Test songs: ext_<name>.npy next to the sid files.  Restartable (skips songs whose outputs all exist).

Layer numbering (documented, keep it consistent):
  mert*              : L<k> = transformers hidden_states[k]  (0 = encoder input, k = output of block k; MERT-330M has 24 blocks;
                       note: the last one is after the final LayerNorm, as returned by HF).
  muscriptor_*       : L<k> = output of model._model.transformer.layers[k]  (0-based block index, 24 blocks in medium / 14 in small).
Pinned, reviewed revisions: see SPECS.   Code reviewed: m-a-p modeling_MERT.py/configuration_MERT.py, muscriptor git 8a1ef88.

python tools/extract_feats.py --encoder mert330 --layers 6 --all --test --gpu 1 [--wait-free]
python tools/extract_feats.py --encoder muscriptor_medium --layers 1,3,5 --probe --out-root work/feats/_probe
python tools/extract_feats.py --selftest --gpu 1
"""
import os, sys, io, time, json, math, argparse, contextlib, datetime, threading, collections
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))       # reviewed muscriptor source (git 8a1ef88), imported as `muscriptor`
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

SPECS = {
    "mert330": dict(kind="mert", repo="m-a-p/MERT-v1-330M", rev="5240c2708a5acaee1007f43fb9735c7dcd0b78c9", sr=24000, fps=75.0, dim=1024, nl=24),
    "mert95": dict(kind="mert", repo="m-a-p/MERT-v1-95M", rev="12af15fef9d0ac838c3f475bfbbf26d2060dd4f5", sr=24000, fps=75.0, dim=768, nl=12),
    "muscriptor_medium": dict(kind="musc", repo="MuScriptor/muscriptor-medium", rev="f32236969308476e01fd3aae67357de5feb05a2d", sr=16000, fps=100.0, dim=1024, nl=24),
    "muscriptor_small": dict(kind="musc", repo="MuScriptor/muscriptor-small", rev="8c127f603b807520fa465c838e9bfee8a91ada4e", sr=16000, fps=100.0, dim=768, nl=14),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _hf_token():
    p = os.path.join(ROOT, "hf_token.txt")
    if os.path.exists(p):
        t = open(p, encoding="utf-8").read().strip()
        if t:
            os.environ["HF_TOKEN"] = t          # never printed / logged
            return t
    return None


def _clip16(x):
    """float32 -> float16 with saturation; returns (array, n_clipped)"""
    big = np.abs(x) > 65000
    n = int(big.sum())
    if n:
        x = np.clip(x, -65000, 65000)
    return x.astype(np.float16), n


# ---------------------------------------------------------------------------------------------- MERT
class MertEncoder:
    WIN = 240000      # 10 s @ 24 kHz
    HOP = 120000      # 5 s
    STRIDE = 320      # samples per frame (75 Hz)
    PAD = 200         # half receptive field (400) -> frame j is centred at j*320 samples of the window
    NW = 751          # frames per window
    KEEP_HI = 562     # keep window frames [187, 562) (first window [0, 562)): >= 2.5 s of context each side
    KEEP_LO = 187
    TILE = 375

    def __init__(self, name, device):
        import torch
        from transformers import AutoModel
        self.name, self.spec, self.device = name, SPECS[name], torch.device(device)
        self.fps, self.sr, self.dim = self.spec["fps"], self.spec["sr"], self.spec["dim"]
        _hf_token()
        m = AutoModel.from_pretrained(self.spec["repo"], revision=self.spec["rev"], trust_remote_code=True)
        self.model = m.float().to(self.device).eval()
        from huggingface_hub import hf_hub_download
        pc = json.load(open(hf_hub_download(self.spec["repo"], "preprocessor_config.json", revision=self.spec["rev"]), encoding="utf-8"))
        self.normalize = bool(pc.get("do_normalize", False))
        log(f"[{name}] loaded, do_normalize={self.normalize}, params={sum(p.numel() for p in m.parameters()) / 1e6:.0f}M")

    def n_frames(self, n_samples):
        return n_samples // self.STRIDE + 1

    def extract(self, wave, layers, batch=4):
        import torch
        wave = np.asarray(wave, dtype=np.float32)
        if self.normalize:
            wave = (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)
        N = len(wave)
        nfr = self.n_frames(N)
        K = 1
        while self.TILE * (K - 1) + self.KEEP_HI < nfr:
            K += 1
        buf = np.zeros(self.HOP * (K - 1) + self.WIN + 2 * self.PAD, dtype=np.float32)
        buf[self.PAD:self.PAD + N] = wave
        out = {l: np.zeros((nfr, self.dim), dtype=np.float16) for l in layers}
        nclip = 0
        with torch.inference_mode():
            for b0 in range(0, K, batch):
                ks = list(range(b0, min(K, b0 + batch)))
                x = torch.from_numpy(np.stack([buf[self.HOP * k: self.HOP * k + self.WIN + 2 * self.PAD] for k in ks])).to(self.device)
                hs = self.model(x, output_hidden_states=True).hidden_states
                assert hs[0].shape[1] == self.NW, hs[0].shape
                for bi, k in enumerate(ks):
                    lo = 0 if k == 0 else self.KEEP_LO
                    g0 = self.TILE * k + lo
                    hi = min(self.KEEP_HI, nfr - self.TILE * k)
                    if hi <= lo:
                        continue
                    for l in layers:
                        a, n = _clip16(hs[l][bi, lo:hi].float().cpu().numpy())
                        out[l][g0:g0 + (hi - lo)] = a
                        nclip += n
        self.last_nclip = nclip
        return out


# ---------------------------------------------------------------------------------------------- MuScriptor
class _Stop(Exception):
    pass


class MuscEncoder:
    SEG = 80000        # 5 s @ 16 kHz (model's training segment)
    HOP = 40000        # 2.5 s stride
    NF = 500           # frames kept per chunk (the 501st mel frame is masked by the model itself)
    HOPF = 250
    STRIDE = 160

    def __init__(self, name, device):
        import torch
        from huggingface_hub import hf_hub_download
        self.name, self.spec, self.device = name, SPECS[name], torch.device(device)
        self.fps, self.sr, self.dim = self.spec["fps"], self.spec["sr"], self.spec["dim"]
        tok = _hf_token()
        r, rev = self.spec["repo"], self.spec["rev"]
        cfgp = hf_hub_download(r, "config.json", revision=rev, token=tok)
        wp = hf_hub_download(r, "model.safetensors", revision=rev, token=tok)        # config.json sits next to it -> read by the loader
        from muscriptor import TranscriptionModel
        self.tm = TranscriptionModel.load_model(weights_path=wp, device=self.device, dtype="float32")
        self.lm = self.tm._model
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        log(f"[{name}] loaded, layers={len(self.lm.transformer.layers)}, params={sum(p.numel() for p in self.lm.parameters()) / 1e6:.0f}M")

    def n_frames(self, n_samples):
        return n_samples // self.STRIDE + 1

    def _conds(self, chunks):
        import torch
        conds = [self.tm._build_conditions(c.unsqueeze(0))[0] for c in chunks]     # c: [80000] -> [1, T]
        with contextlib.redirect_stdout(io.StringIO()):                              # conditioner prints timing lines
            prepared = self.lm.condition_provider.tokenize(conds)
            return self.lm.condition_provider(prepared)

    def hidden(self, chunks, layers):
        """chunks: list of float32 tensors [80000] (on device). -> {layer: [B, 500, D]} (single first_step forward, audio positions)"""
        import torch
        B = len(chunks)
        ct = self._conds(chunks)
        seq = torch.full((B, 1), self.lm.initial_token_id, dtype=torch.long, device=self.device)
        caps, hooks = {}, []
        kmax = max(layers)

        def mk(k):
            def h(mod, inp, outp):
                caps[k] = outp[:, :self.NF].detach()
                if k == kmax:
                    raise _Stop()
            return h
        for k in layers:
            hooks.append(self.lm.transformer.layers[k].register_forward_hook(mk(k)))
        try:
            self.lm(seq, ct, first_step=True)
        except _Stop:
            pass
        finally:
            for h in hooks:
                h.remove()
        return caps

    def extract(self, wave, layers, batch=8):
        import torch
        wave = np.asarray(wave, dtype=np.float32)
        N = len(wave)
        nfr = self.n_frames(N)
        K = 1
        while self.HOPF * (K - 1) + self.NF < nfr:
            K += 1
        buf = np.zeros(self.HOP * (K - 1) + self.SEG, dtype=np.float32)
        buf[:N] = wave
        out = {l: np.zeros((nfr, self.dim), dtype=np.float16) for l in layers}
        nclip = 0
        with torch.inference_mode():
            for b0 in range(0, K, batch):
                ks = list(range(b0, min(K, b0 + batch)))
                chunks = [torch.from_numpy(buf[self.HOP * k: self.HOP * k + self.SEG]).to(self.device) for k in ks]
                caps = self.hidden(chunks, layers)
                for bi, k in enumerate(ks):
                    lo = 0 if k == 0 else self.HOPF            # causal attention: keep the later half of every chunk
                    g0 = self.HOPF * k + lo
                    hi = min(self.NF, nfr - self.HOPF * k)
                    if hi <= lo:
                        continue
                    for l in layers:
                        a, n = _clip16(caps[l][bi, lo:hi].float().cpu().numpy())
                        out[l][g0:g0 + (hi - lo)] = a
                        nclip += n
        self.last_nclip = nclip
        return out


def load_encoder(name, device="cuda:0"):
    return (MertEncoder if SPECS[name]["kind"] == "mert" else MuscEncoder)(name, device)


def extract(wave, encoder, layers, batch=None):
    """wave: float32 mono numpy at encoder.sr; encoder: object from load_encoder(); layers: list of ints (see module docstring)
    -> {layer: float16 [T, D]}, frame i at time i / encoder.fps"""
    return encoder.extract(wave, list(layers), **({} if batch is None else {"batch": batch}))


# ---------------------------------------------------------------------------------------------- driver
def out_path(root, enc, layer, key):
    return os.path.join(root, enc, f"L{layer}", f"{key}.npy")


def items_for(args):
    meta = json.load(open(os.path.join(ROOT, "work", "meta.json"), encoding="utf-8"))
    songs = {s[0]: s for s in meta["songs"]}
    items = []
    if args.probe:
        pj = json.load(open(os.path.join(ROOT, "work", "feats", "_probe", "songs.json")))
        sids = pj["train"] + pj["val"]
    elif args.sids:
        sids = [int(x) for x in args.sids.split(",")]
    elif args.all:
        sids = sorted(songs)
    else:
        sids = []
    for s in sids:
        items.append((f"{s:05d}", os.path.join(ROOT, "official charts", songs[s][1], "track.mp3"), songs[s][2]))
    if args.test:
        td = os.path.join(ROOT, "test_songs")
        for f in sorted(os.listdir(td)):
            if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                items.append((f"ext_{os.path.splitext(f)[0]}", os.path.join(td, f), None))
    if args.limit:
        items = items[:args.limit]
    return items


def wait_free():
    """full extraction on cuda:1 only when no 7xx job runs, or after 18:00 local time"""
    while True:
        running = [f for f in os.listdir(os.path.join(ROOT, "jobs", "running")) if f.startswith("7") and f.endswith(".cmd")]
        if not running or datetime.datetime.now().hour >= 18:
            return
        log(f"waiting: 7xx jobs running {running}, local hour {datetime.datetime.now().hour}")
        time.sleep(120)


def run(args):
    from mai import audio
    items = items_for(args)
    if args.layers == "best":      # chosen by tools/probe_feats.py report
        layers = [json.load(open(os.path.join(ROOT, "work", "feats", "_probe", "best.json")))[args.encoder]]
    else:
        layers = [int(x) for x in args.layers.split(",")]
    root = args.out_root
    todo = [it for it in items if not all(os.path.exists(out_path(root, args.encoder, l, it[0])) for l in layers)]
    log(f"{args.encoder} layers {layers}: {len(items)} items, {len(todo)} to do -> {root}")
    if not todo:
        return
    if args.wait_free:
        wait_free()
    enc = load_encoder(args.encoder, "cuda:0")
    sr = enc.sr
    tot_audio = tot_time = 0.0
    tot_clip = 0
    t_start = time.time()
    total_dur = sum((it[2] or 150.0) for it in todo)
    done_dur = 0.0
    with ThreadPoolExecutor(1) as ex:
        futs = collections.deque()
        it_iter = iter(todo)

        def fill():
            while len(futs) < 2:
                try:
                    it = next(it_iter)
                except StopIteration:
                    return
                futs.append((it, ex.submit(audio.load_audio, it[1], sr)))
        fill()
        i = 0
        while futs:
            it, fut = futs.popleft()
            fill()
            key = it[0]
            try:
                wave = fut.result()
                t0 = time.time()
                res = extract(wave, enc, layers, args.batch)
                import torch
                torch.cuda.synchronize()
                dt = time.time() - t0
                for l in layers:
                    p = out_path(root, args.encoder, l, key)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p + ".tmp", "wb") as f:
                        np.save(f, res[l])
                    os.replace(p + ".tmp", p)
                dur = len(wave) / sr
                tot_audio += dur; tot_time += dt; done_dur += (it[2] or dur)
                tot_clip += enc.last_nclip
                i += 1
                if i % args.log_every == 0 or i == len(todo) or i <= 3:
                    el = time.time() - t_start
                    eta = el / max(done_dur, 1) * (total_dur - done_dur)
                    log(f"[{i}/{len(todo)}] {key} {dur:.0f}s audio, {dt:.1f}s gpu (x{dur / max(dt, 1e-6):.1f} realtime; mean x{tot_audio / tot_time:.1f}), "
                        f"elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min, clipped {tot_clip}")
            except Exception as e:  # noqa
                import traceback
                log(f"FAILED {key}: {e!r}")
                traceback.print_exc()
    log(f"DONE {args.encoder}: {tot_audio / 3600:.2f} h audio in {tot_time / 60:.1f} min gpu (x{tot_audio / max(tot_time, 1e-6):.1f} realtime); wall {(time.time() - t_start) / 60:.1f} min; fp16-saturated values {tot_clip}")


def selftest(args):
    """small checks on cuda:1: speed, overflow, window/chunk consistency, generate()-equivalence of the hook forward"""
    import torch
    from mai import audio
    meta = json.load(open(os.path.join(ROOT, "work", "meta.json"), encoding="utf-8"))
    s = meta["songs"][5]
    path = os.path.join(ROOT, "official charts", s[1], "track.mp3")
    for name in args.encoders.split(","):
        spec = SPECS[name]
        wave = audio.load_audio(path, spec["sr"])[: spec["sr"] * 40]       # 40 s
        t0 = time.time()
        enc = load_encoder(name, "cuda:0")
        log(f"{name}: load {time.time() - t0:.1f}s, wave {len(wave) / spec['sr']:.1f}s")
        L = [1, 5, 11] if spec["kind"] == "musc" else [2, 6, 12]
        for rep in range(2):
            torch.cuda.synchronize(); t0 = time.time()
            res = extract(wave, enc, L)
            torch.cuda.synchronize(); dt = time.time() - t0
            log(f"  extract {len(wave) / spec['sr']:.0f}s audio: {dt:.2f}s -> x{len(wave) / spec['sr'] / dt:.1f} realtime, clipped {enc.last_nclip}, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        for l in L:
            a = res[l].astype(np.float32)
            log(f"  L{l}: shape {a.shape} (expect T={enc.n_frames(len(wave))}), absmax {np.abs(a).max():.1f}, std {a.std():.3f}, nan {np.isnan(a).sum()}, zero-frames {(np.abs(a).sum(1) == 0).sum()}")
        # consistency: shifted extraction (drop first 1.37 s, a non-multiple of the tiling) vs original
        shift_s = 1.37 if spec["kind"] == "musc" else 1.28          # MERT: multiple of 320 samples (1.28 s = 96 frames)
        sh = int(round(shift_s * spec["sr"]))
        sh -= sh % 320 if spec["kind"] == "mert" else sh % 160
        res2 = extract(wave[sh:], enc, L)
        off = int(round(sh / (spec["sr"] / enc.fps)))
        for l in L:
            a = res[l][off: off + 2500].astype(np.float32); b = res2[l][: len(a)].astype(np.float32)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            log(f"  L{l}: cos(original, shifted-by-{sh / spec['sr']:.2f}s) mean {cos.mean():.4f} p5 {np.percentile(cos, 5):.4f} min {cos.min():.4f}")
        if spec["kind"] == "musc":
            # hook forward == the first decoding step of the library's own generate()
            chunk = torch.from_numpy(wave[: enc.SEG]).to(enc.device)
            ct = enc._conds([chunk])
            seq = torch.full((1, 1), enc.lm.initial_token_id, dtype=torch.long, device=enc.device)
            with torch.inference_mode():
                lg = enc.lm(seq, ct, first_step=True)[0, -1].float()
                lg[1393:] = -float("inf")
                mine = int(lg.argmax())
                with contextlib.redirect_stdout(io.StringIO()):
                    first = next(iter(enc.lm.generate(conditions=[enc.tm._build_conditions(chunk.unsqueeze(0))[0]], max_gen_len=2, use_sampling=False, cfg_coef=1.0)))
            log(f"  first-token check: mine {mine}, generate() {int(first[0])}")
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="mert330", choices=list(SPECS))
    ap.add_argument("--layers", default="6")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--test", action="store_true", help="also the test_songs/ mp3s -> ext_<name>.npy")
    ap.add_argument("--sids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "work", "feats"))
    ap.add_argument("--gpu", default="1", help="physical GPU index (PCI bus order); exposed as cuda:0 inside the process")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--wait-free", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--encoders", default="mert330,muscriptor_medium")
    a = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu            # never touches GPU 0 (training)
    import torch
    torch.set_num_threads(2)
    if a.batch == 0:
        a.batch = None
    if a.selftest:
        selftest(a)
    else:
        if a.batch is None:
            a.batch = 4 if SPECS[a.encoder]["kind"] == "mert" else 8
        run(a)

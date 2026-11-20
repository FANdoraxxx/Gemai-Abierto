"""Frozen pretrained audio-encoder features (see tools/extract_feats.py) -> model inputs.

Files: work/feats/<encoder>/L<layer>/<sid:05d>.npy (or ext_<name>.npy for test songs), float16 [T, D], frame i at time i / fps.
Encoders / fps: mert330 75 Hz D=1024, mert95 75 Hz D=768, muscriptor_medium 100 Hz D=1024, muscriptor_small 100 Hz D=768.
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS_ROOT = os.environ.get("MAI_FEATS_ROOT") or os.path.join(ROOT, "work", "feats")   # e.g. an SSD copy: set MAI_FEATS_ROOT=E:\\maiAIV3_feats
# v5: fan (self-made chart) songs live in their own id space: sid = FAN_SID0 + aid.  Their MERT features are at
# $MAI_FAN_FEATS_ROOT (default <project>/fan/feats) under the aid-numbered file name; official songs are untouched.
FAN_SID0 = 100000
FAN_FEATS_FALLBACK = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "fan", "feats")   # the G: originals
FAN_FEATS_ROOT = os.environ.get("MAI_FAN_FEATS_ROOT") or FAN_FEATS_FALLBACK   # SSD cache may be partial: missing files fall back per file
FPS = {"mert330": 75.0, "mert95": 75.0, "muscriptor_medium": 100.0, "muscriptor_small": 100.0}
DIM = {"mert330": 1024, "mert95": 768, "muscriptor_medium": 1024, "muscriptor_small": 768}


def feats_path(encoder, layer, sid_or_name, root=None):
    """sid (int) -> <sid:05d>.npy ; str name -> ext_<name>.npy (test songs; a leading 'ext_' is accepted)"""
    if isinstance(sid_or_name, (int, np.integer)):
        if int(sid_or_name) >= FAN_SID0 and root is None:
            rel = os.path.join(encoder, f"L{layer}", f"{int(sid_or_name) - FAN_SID0:05d}.npy")
            p = os.path.join(FAN_FEATS_ROOT, rel)
            return p if os.path.exists(p) else os.path.join(FAN_FEATS_FALLBACK, rel)
        key = f"{int(sid_or_name):05d}"
    else:
        key = str(sid_or_name)
        if key.isdigit():
            key = f"{int(key):05d}"
        elif not key.startswith("ext_"):
            key = "ext_" + key
    return os.path.join(root or FEATS_ROOT, encoder, f"L{layer}", key + ".npy")


def load_feats(encoder, layer, sid_or_name, root=None):
    """memory-mapped float16 [T, D]; frame i is centred at time i / FPS[encoder] seconds (audio time, 0 = start of track.mp3)"""
    return np.load(feats_path(encoder, layer, sid_or_name, root), mmap_mode="r")


def best_layer(encoder):
    """layer chosen by the probe (work/feats/_probe/best.json) -- or pass the layer explicitly"""
    return int(json.load(open(os.path.join(FEATS_ROOT, "_probe", "best.json")))[encoder])


def _beat_sync_feats_ref(feats, fps, times_sec):
    """straightforward reference implementation (kept for equivalence tests)"""
    T = feats.shape[0]
    f = np.asarray(times_sec, dtype=np.float64) * fps
    i0 = np.floor(f).astype(np.int64)
    w = (f - i0).astype(np.float32)[:, None]
    valid = (i0 >= 0) & (i0 + 1 < T)
    i0c = np.clip(i0, 0, T - 2)
    a = np.asarray(feats[i0c], dtype=np.float32)
    b = np.asarray(feats[i0c + 1], dtype=np.float32)
    return a * (1 - w) + b * w, valid


def beat_sync_feats(feats, fps, times_sec):
    """Mirror of mai.audio.beat_sync: feats [T, D] (any fps, e.g. the np.load mmap), times_sec [F] -> ([F, D] float32 linear interpolation, valid [F] bool).
    Frames outside [0, T-2+1) are not valid (values there are clamped copies of the edge frames; callers mask them as for mel).
    Identical result to _beat_sync_feats_ref, but reads ONE contiguous slice of the (mmap) array first, then gathers in RAM."""
    T = feats.shape[0]
    f = np.asarray(times_sec, dtype=np.float64) * fps
    i0 = np.floor(f).astype(np.int64)
    w = (f - i0).astype(np.float32)[:, None]
    valid = (i0 >= 0) & (i0 + 1 < T)
    i0c = np.clip(i0, 0, T - 2)
    lo, hi = int(i0c.min()), int(i0c.max()) + 2
    sl = np.asarray(feats[lo:hi])                    # one sequential read (fp16)
    i0c -= lo
    a = np.take(sl, i0c, axis=0).astype(np.float32)
    i0c += 1
    b = np.take(sl, i0c, axis=0).astype(np.float32)
    b -= a
    b *= w
    a += b                                           # a + (b - a) * w  == a * (1 - w) + b * w up to fp32 rounding
    return a, valid


def window_slice_feats(feats, fps, times_sec):
    """For GPU-side interpolation (recommended for training, see docs/FEATS_INTEGRATION.md).
    -> (sl float16 [n, D] contiguous native frames, i0 int32 [F] index into sl, w float32 [F] weight, valid [F] bool)
    value at times_sec[j] = sl[i0[j]] * (1 - w[j]) + sl[i0[j] + 1] * w[j]   (== beat_sync_feats, exactly; i0 + 1 <= n - 1 always)
    Invalid frames (outside the song) have clamped i0 / w and must be masked by the caller."""
    T = feats.shape[0]
    f = np.asarray(times_sec, dtype=np.float64) * fps
    i0 = np.floor(f).astype(np.int64)
    w = (f - i0).astype(np.float32)
    valid = (i0 >= 0) & (i0 + 1 < T)
    i0c = np.clip(i0, 0, T - 2)
    lo, hi = int(i0c.min()), int(i0c.max()) + 2
    sl = np.array(feats[lo:hi])                      # real copy of one contiguous range (reads the mmap now)
    return sl, (i0c - lo).astype(np.int32), w, valid


def interp_pool_torch(sl, i0, w, valid, group=4):
    """GPU/CPU torch twin of beat_sync_feats followed by a box average over `group` consecutive grid frames (4 ticks = one encoder token).
    sl [B, N, D] (fp16/fp32, zero padded to the longest slice in the batch), i0 [B, F] int (index into sl), w [B, F] float, valid [B, F] bool
    (F = 1728 ticks; i0 + 1 <= N - 1).  -> (tokens [B, F // group, D] float32, tok_valid [B, F // group] bool = any valid frame in the group).
    Tokens with no valid frame are exactly 0 (replace them by a learned 'invalid' vector in the model)."""
    import torch
    B, N, D = sl.shape
    F_ = i0.shape[1]
    ix = i0.long().unsqueeze(-1).expand(-1, -1, D)
    a = torch.gather(sl, 1, ix).float()
    b = torch.gather(sl, 1, ix + 1).float()
    v = torch.lerp(a, b, w.float().unsqueeze(-1))
    m = valid.float().view(B, F_ // group, group, 1)
    v = (v.view(B, F_ // group, group, D) * m).sum(2) / m.sum(2).clamp(min=1.0)
    return v, valid.view(B, F_ // group, group).any(-1)

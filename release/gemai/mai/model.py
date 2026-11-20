"""ChartModel: mel encoder + causal decoder with cross-attention locality bias (see docs/MODEL_SPEC.md)."""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

TPB = 48


@dataclass
class ModelConfig:
    vocab_size: int = 0
    d: int = 512
    heads: int = 8
    enc_layers: int = 6
    dec_layers: int = 8
    ffn: int = 2048
    dropout: float = 0.1
    max_len: int = 1024
    n_mels: int = 96
    audio_frames: int = 1728     # mel frames per window, 1/48 beat each (ticks -768 .. +959)
    t_start: int = -768          # tick (relative to window start) of audio frame 0
    n_phase: int = 48
    n_beat: int = 40
    beat_offset: int = 16
    # ---- seq_version 2 additions (all defaults reproduce the v1 model exactly: no extra parameters are created)
    seq_version: int = 1
    attn_dropout: float = -1.0   # < 0: same as `dropout`
    lv_cont: bool = False        # smooth level embedding (Fourier features of the level) added to the LV_k token embedding
    cond_inject: bool = False    # sum of the condition-token embeddings is added to the input of EVERY decoder position
    n_cond: int = 8              # number of condition tokens after BOS (8 for v1, 9 for v2: + DENS)
    lv0: int = 0                 # vocab id of LV_0 (LV_k = lv0 + k  <->  level 1.0 + 0.1 k)
    n_lv: int = 141
    # ---- seq_version 5: BD1..BD4 header tokens (per-bar note-count buckets).  bd0 = vocab id of BD1_UNK, n_bd = number of consecutive BD ids; n_bd == 0 -> off
    bd0: int = 0
    n_bd: int = 0
    # ---- optional fusion of frozen pretrained audio features (mai/feats.py); feat_encoder == "" -> off, no extra parameters
    feat_encoder: str = ""       # e.g. "mert330"
    feat_layer: int = 0          # e.g. 8
    feat_dim: int = 0            # channels of the stored features (1024)
    feat_fps: float = 0.0        # frame rate of the stored features (75 for mert330)
    feat_p_drop: float = 0.15    # training: whole feature branch off for a sample (mel-only fallback)
    feat_p_chan: float = 0.10    # training: dropout on the projected feature vector
    feat_p_span: float = 0.30    # training: probability that 1-3 short time spans of the feature tokens are zeroed
    # ---- seq_version 3: song-level context.  song_dim == 0 -> off (no extra parameters; v1/v2/v3a checkpoints behave identically)
    song_dim: int = 0            # width of the song encoder (0 = off)
    song_layers: int = 4         # bidirectional transformer layers over the per-bar tokens
    song_heads: int = 4
    song_ffn: int = 1024
    song_xattn: str = "1,3,5,7"  # decoder layers (0-based) that cross-attend to the song memory
    song_in: int = 1218          # columns of the bar table (mai/v3.py: 1024 MERT + 96 mel mean + 96 mel std + flux + valid fraction)
    song_mert: int = 1024
    song_idx: int = 256          # absolute bar-index embedding size
    song_rel_n: int = 97         # relative-bar embedding size (mai.v3.rel_index)
    song_p_drop: float = 0.10    # training: the whole song memory is switched off for a sample (fallback path stays trained)
    song_p_bar: float = 0.10     # training: a bar token is replaced by the learned "missing bar" vector
    # ---- seq_version 4: LLM structure annotations as extra per-bar inputs of the song encoder (mai/ann.py).  song_ann False -> no extra parameters
    song_ann: bool = False
    song_ann_std: float = 0.1    # init std of the annotation embeddings (row 0 = UNK starts at 0: an unannotated song == the v3 model at step 0)
    ann_n_sec: int = 10          # labels (+ UNK row 0)
    ann_n_pos: int = 10          # bars-since-section-start buckets (+ UNK)
    ann_n_end: int = 5           # bars-left-in-section buckets (+ UNK)
    ann_n_flg: int = 5           # boundary / last bar / big_hit / riser / drop
    # ---- v6 infill: FUT marker token in the vocabulary (no extra parameters; only the embedding table is one row larger)
    with_fut: bool = False

    @property
    def enc_len(self):
        return self.audio_frames // 4

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _inv_softplus(x):
    return math.log(math.expm1(x))


def lv_features(lv):
    """smooth features of the level (lv in [1, 15]): z, z^2 (z = (lv-12)/3) and sin/cos with periods 3, 6, 12 levels"""
    z = (lv - 12.0) / 3.0
    f = [z, z * z]
    for period in (3.0, 6.0, 12.0):
        w = 2.0 * math.pi / period
        f += [torch.sin(w * lv), torch.cos(w * lv)]
    return torch.stack(f, -1)


def _autocast_dtype(device_type):
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except Exception:
        pass
    return None


class FeatBranch(nn.Module):
    """frozen pretrained features [B, 432, D] (interpolated onto the mel grid, 4-tick box average) -> [B, 432, d], added to the conv-mel tokens.
    LayerNorm first (MERT values have std ~10 with outlier channels)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.feat_dim)
        self.lin = nn.Linear(cfg.feat_dim, cfg.d)          # (not called `proj`: ChartModel rescales *proj.weight inits)
        self.chan = nn.Dropout(cfg.feat_p_chan)
        self.invalid = nn.Parameter(torch.zeros(cfg.d))
        self.scale = nn.Parameter(torch.ones(()))
        self.p_drop, self.p_span = cfg.feat_p_drop, cfg.feat_p_span

    def forward(self, feats):
        """feats: dict(fsl [B,N,D], fi0 [B,F], fw [B,F], fvalid [B,F], optional keep [B] bool) -> [B, 432, d]"""
        from .feats import interp_pool_torch
        dev = self.lin.weight.device
        tok, tv = interp_pool_torch(feats["fsl"].to(dev), feats["fi0"].to(dev), feats["fw"].to(dev), feats["fvalid"].to(dev), 4)
        B, S, _ = tok.shape
        x = self.lin(self.ln(tok))
        x = self.chan(x)
        x = torch.where(tv.unsqueeze(-1), x, self.invalid.to(x.dtype))
        keep = feats.get("keep")
        keep = torch.ones(B, dtype=torch.bool, device=dev) if keep is None else keep.to(dev)
        if self.training:
            keep = keep & (torch.rand(B, device=dev) >= self.p_drop)
            if self.p_span > 0:                             # SpecAugment-like: up to 3 spans of 2-12 tokens (8-48 ticks) zeroed
                has = torch.rand(B, device=dev) < self.p_span
                nsp = torch.randint(1, 4, (B,), device=dev)
                ar = torch.arange(S, device=dev)[None]
                zero = torch.zeros(B, S, dtype=torch.bool, device=dev)
                for j in range(3):
                    act = (has & (nsp > j))[:, None]
                    st = torch.randint(0, S, (B, 1), device=dev)
                    ln = torch.randint(2, 13, (B, 1), device=dev)
                    zero |= act & (ar >= st) & (ar < st + ln)
                x = x.masked_fill(zero.unsqueeze(-1), 0.0)
        return self.scale * x * keep[:, None, None].to(x.dtype)


class SongBlock(nn.Module):
    """pre-LN bidirectional transformer block of the song encoder (bool key-padding mask [B,1,1,S])"""

    def __init__(self, ds, heads, ffn, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(ds), nn.LayerNorm(ds)
        self.attn = SelfAttention(ds, heads, dropout, causal=False)
        self.ff = FFN(ds, ffn)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.ff(self.ln2(x)))


class AnnEmbed(nn.Module):
    """per-bar annotation features (mai/ann.py bar_arrays) -> [B,S,ds] added to the bar-token inputs of the song encoder.
    Index 0 of every embedding is the explicit UNK ("no annotation") row; it starts at 0 and is learned."""

    def __init__(self, cfg):
        super().__init__()
        ds = cfg.song_dim
        self.lab = nn.Embedding(cfg.ann_n_sec + 1, ds)
        self.pos = nn.Embedding(cfg.ann_n_pos + 1, ds)
        self.end = nn.Embedding(cfg.ann_n_end + 1, ds)
        self.flg = nn.Linear(cfg.ann_n_flg, ds, bias=False)
        self.std = cfg.song_ann_std
        self.reset()

    def reset(self):
        for e in (self.lab, self.pos, self.end):
            nn.init.normal_(e.weight, std=self.std)
            with torch.no_grad():
                e.weight[0].zero_()
        nn.init.normal_(self.flg.weight, std=self.std)

    def forward(self, alab, apos, aend, aflg):
        return self.lab(alab) + self.pos(apos) + self.end(aend) + self.flg(aflg.to(self.flg.weight.dtype))


class SongEncoder(nn.Module):
    """per-bar summaries of the WHOLE song (mai/v3.py bar table, <= 160 tokens) -> contextualised memory + auxiliary intensity head.
    Depends on the audio only (never on chart tokens)."""

    def __init__(self, cfg):
        super().__init__()
        ds, nm = cfg.song_dim, cfg.song_in - cfg.song_mert - 1
        self.n_mert, self.n_mel = cfg.song_mert, nm
        self.ln_m, self.p_m = nn.LayerNorm(cfg.song_mert), nn.Linear(cfg.song_mert, ds)
        self.ln_l, self.p_l = nn.LayerNorm(nm), nn.Linear(nm, ds)
        self.p_s = nn.Linear(5, ds)                      # [log bpm, relative position, pool factor, n_bars, valid fraction]
        self.idx_emb = nn.Embedding(cfg.song_idx, ds)
        self.missing = nn.Parameter(torch.zeros(ds))
        self.blocks = nn.ModuleList([SongBlock(ds, cfg.song_heads, cfg.song_ffn, cfg.dropout) for _ in range(cfg.song_layers)])
        self.ln = nn.LayerNorm(ds)
        self.aux = nn.Sequential(nn.Linear(ds, ds), nn.GELU(), nn.Linear(ds, 4))   # log1p(notes/bar), hold / slide / touch share (logits)
        self.drop = nn.Dropout(cfg.dropout)
        self.p_bar = cfg.song_p_bar
        self.ann = AnnEmbed(cfg) if cfg.song_ann else None

    def forward(self, sfeat, sscal, sidx, smask, ann=None):
        """sfeat [B,S,song_in] fp16, sscal [B,S,4], sidx [B,S] long, smask [B,S] bool (True = real bar) -> (memory [B,S,ds], aux [B,S,4]).
        ann: None | dict(alab, apos, aend [B,S] long, aflg [B,S,5]) (a model with song_ann and ann=None sees every bar as UNK)"""
        x = sfeat.float()                                # MERT values reach 300+: never square them in fp16
        vf = x[..., -1:]
        h = self.p_m(self.ln_m(x[..., :self.n_mert])) + self.p_l(self.ln_l(x[..., self.n_mert:-1])) \
            + self.p_s(torch.cat([sscal.float(), vf], -1)) + self.idx_emb(sidx)
        if self.ann is not None:
            if ann is None:
                z = torch.zeros_like(sidx)
                ann = dict(alab=z, apos=z, aend=z, aflg=torch.zeros(*sidx.shape, self.ann.flg.in_features, device=sidx.device))
            h = h + self.ann(ann["alab"], ann["apos"], ann["aend"], ann["aflg"])
        miss = vf <= 0.05                                # bar beyond the end of the audio
        if self.training and self.p_bar > 0:
            miss = miss | (torch.rand(h.shape[0], h.shape[1], 1, device=h.device) < self.p_bar)
        h = torch.where(miss, self.missing.to(h.dtype) + self.idx_emb(sidx), h)
        h = self.drop(h)
        am = smask[:, None, None, :]
        for blk in self.blocks:
            h = blk(h, am)
        h = self.ln(h)
        return h, self.aux(h)


class SongCrossAttention(nn.Module):
    """decoder -> song memory cross-attention (no locality bias: the memory already carries each bar's position relative to the window)"""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.hd, self.p = heads, d // heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def make_kv(self, mem):
        B, S, _ = mem.shape
        k, v = self.kv(mem).view(B, S, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, k, v, mask, keep):
        """x [B,L,d]; k, v [B,H,S,hd]; mask bool [B,1,1,S] (True = attend); keep [B] float (0 = this sample has no song context)"""
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask,
                                           dropout_p=self.p if self.training else 0.0)
        y = self.proj(y.transpose(1, 2).reshape(B, L, D))
        return y * keep.to(y.dtype).view(B, 1, 1)


def song_aux_loss(aux, target, mask):
    """auxiliary intensity loss on the song encoder.  aux [B,S,4]; target [B,S,5] (mai.v3.AUX_COLS); mask [B,S] bool.
    smooth-L1 on log1p(mean notes per bar) + 0.5 * BCE of the hold / slide / touch shares (only where the token has notes).
    Returns the mean over valid tokens (0-d tensor)."""
    m = mask.float()
    reg = F.smooth_l1_loss(aux[..., 0].float(), target[..., 0].float(), reduction="none", beta=1.0)
    sh = F.binary_cross_entropy_with_logits(aux[..., 1:4].float(), target[..., 1:4].float(), reduction="none").mean(-1) * target[..., 4].float()
    return ((reg + 0.5 * sh) * m).sum() / m.sum().clamp(min=1.0)


class FFN(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1 = nn.Linear(d, f)
        self.w2 = nn.Linear(f, d)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class SelfAttention(nn.Module):
    def __init__(self, d, heads, dropout, causal):
        super().__init__()
        self.h, self.hd, self.p, self.causal = heads, d // heads, dropout, causal
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def _qkv(self, x):
        B, L, _ = x.shape
        return self.qkv(x).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)  # 3 x [B,H,L,hd]

    def forward(self, x, mask=None, kv_out=None):
        B, L, D = x.shape
        q, k, v = self._qkv(x)
        if kv_out is not None:                       # prefill: store K/V into the cache
            kv_out[0][:, :, :L] = k
            kv_out[1][:, :, :L] = v
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0,
            is_causal=(self.causal and mask is None))
        return self.proj(y.transpose(1, 2).reshape(B, L, D))

    def step(self, x, ck, cv, pos):
        B, _, D = x.shape
        q, k, v = self._qkv(x)
        ck[:, :, pos:pos + 1] = k
        cv[:, :, pos:pos + 1] = v
        y = F.scaled_dot_product_attention(q, ck[:, :, :pos + 1].to(q.dtype), cv[:, :, :pos + 1].to(q.dtype))
        return self.proj(y.transpose(1, 2).reshape(B, 1, D))


class CrossAttention(nn.Module):
    """Cross-attention with a learned per-head locality bias on |tau_j - t_i| (in beats)."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.hd, self.p = heads, d // heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        ladder = [0.0, 0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
        if heads == 8:
            vals = ladder
        else:
            n = min(heads, 6)
            vals = [0.0] * (heads - n) + ladder[-n:]
        self.a_fut = nn.Parameter(torch.tensor([_inv_softplus(max(v, 1e-3)) for v in vals]))
        self.a_past = nn.Parameter(torch.tensor([_inv_softplus(max(2 * v, 1e-3)) for v in vals]))

    def make_kv(self, enc):
        B, S, _ = enc.shape
        k, v = self.kv(enc).view(B, S, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        return k, v                                  # [B,H,S,hd]

    def slopes(self):
        return F.softplus(self.a_fut), F.softplus(self.a_past)

    def bias(self, dpos, dneg, dtype):
        """dpos = relu(tau_j - t_i)/48, dneg = relu(t_i - tau_j)/48 : [B,L,S] -> [B,H,L,S]"""
        if self.training or torch.is_grad_enabled():
            af, ap = self.slopes()
            af = (-af).to(dtype).view(1, -1, 1, 1)
            ap = (-ap).to(dtype).view(1, -1, 1, 1)
        else:
            # speed (inference only): the slope tensors depend on parameters alone -> compute once per (dtype, autocast, param version)
            key = (dtype, torch.is_autocast_enabled(), self.a_fut._version, self.a_past._version, self.a_fut.device)
            hit = getattr(self, "_slope_cache", None)
            if hit is None or hit[0] != key:
                af, ap = self.slopes()
                hit = self._slope_cache = (key, (-af).to(dtype).view(1, -1, 1, 1), (-ap).to(dtype).view(1, -1, 1, 1))
            af, ap = hit[1], hit[2]
        return dpos.unsqueeze(1).to(dtype) * af + dneg.unsqueeze(1).to(dtype) * ap

    def forward(self, x, k, v, dpos, dneg):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.hd).transpose(1, 2)
        b = self.bias(dpos, dneg, q.dtype)
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=b,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


def _attn_p(cfg):
    return cfg.dropout if cfg.attn_dropout < 0 else cfg.attn_dropout


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, cfg.heads, _attn_p(cfg), causal=False)
        self.ff = FFN(d, cfg.ffn)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderBlock(nn.Module):
    def __init__(self, cfg, with_song=False):
        super().__init__()
        d = cfg.d
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = SelfAttention(d, cfg.heads, _attn_p(cfg), causal=True)
        self.ca = CrossAttention(d, cfg.heads, _attn_p(cfg))
        self.ff = FFN(d, cfg.ffn)
        self.drop = nn.Dropout(cfg.dropout)
        self.sca = None
        if with_song:                                 # v3: song-memory cross-attention between audio cross-attention and FFN
            self.ln_s = nn.LayerNorm(d)
            self.sca = SongCrossAttention(d, cfg.heads, _attn_p(cfg))

    def forward(self, x, enc, dpos, dneg, mask=None, song=None):
        k, v = self.ca.make_kv(enc)
        x = x + self.drop(self.sa(self.ln1(x), mask))
        x = x + self.drop(self.ca(self.ln2(x), k, v, dpos, dneg))
        if self.sca is not None and song is not None:
            sk, sv = self.sca.make_kv(song["mem"])
            x = x + self.drop(self.sca(self.ln_s(x), sk, sv, song["mask"], song["keep"]))
        return x + self.drop(self.ff(self.ln3(x)))


class DecodeCache:
    def __init__(self, k, v, ck, cv):
        self.k, self.v, self.ck, self.cv = k, v, ck, cv
        self.len = 0
        self.g = None          # [B,d] condition vector (cfg.cond_inject), set by prefill
        self.sk = self.sv = None   # per decoder layer song-memory K/V (v3; None entries for layers without song attention)
        self.smask = self.skeep = None


class ChartModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0
        self.cfg = cfg
        d = cfg.d
        self.conv1 = nn.Conv1d(cfg.n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.enc_pos = nn.Parameter(torch.zeros(cfg.enc_len, d))
        self.enc_blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.enc_layers)])
        self.enc_ln = nn.LayerNorm(d)
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        self.phase_emb = nn.Embedding(cfg.n_phase, d)
        self.beat_emb = nn.Embedding(cfg.n_beat, d)
        self.song_layer_ids = sorted({int(x) for x in cfg.song_xattn.split(",") if x.strip() != ""}) if cfg.song_dim else []
        self.dec_blocks = nn.ModuleList([DecoderBlock(cfg, with_song=(i in self.song_layer_ids)) for i in range(cfg.dec_layers)])
        self.dec_ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.lv_cont:
            self.lv_lin = nn.Linear(8, d)
            self.register_buffer("lv_feat", lv_features(1.0 + 0.1 * torch.arange(cfg.n_lv, dtype=torch.float32)),
                                 persistent=False)
        if cfg.cond_inject:
            self.cond_emb = nn.Embedding(cfg.vocab_size, d)
            self.cond_ln = nn.LayerNorm(d)
            self.cond_out = nn.Linear(d, d)
        if cfg.n_bd:                                      # own small tables (own learning rate in train.py): input embedding + condition-injection embedding of the BD ids
            self.bd_tok = nn.Embedding(cfg.n_bd, d)
            self.bd_cond = nn.Embedding(cfg.n_bd, d)
        if cfg.feat_encoder:
            self.feat = FeatBranch(cfg)
        else:
            self.feat = None
        if cfg.song_dim:
            self.song = SongEncoder(cfg)
            self.song_to_dec = nn.Linear(cfg.song_dim, d)     # (not `*proj`: the proj-weight rescale below must not touch it)
            self.song_rel = nn.Embedding(cfg.song_rel_n, d)
            self.song_ln = nn.LayerNorm(d)
        tau = cfg.t_start + 4 * torch.arange(cfg.enc_len, dtype=torch.float32) + 2
        self.register_buffer("tau", tau, persistent=False)
        self.grad_ckpt = False
        self.apply(self._init)
        nn.init.normal_(self.enc_pos, std=0.02)
        if cfg.cond_inject:
            nn.init.zeros_(self.cond_out.weight)          # start as a plain decoder; the injection is learned
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("ff.w2.weight"):
                n = cfg.enc_layers if name.startswith("enc_blocks") else cfg.dec_layers
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * max(n, 1)))
        if cfg.song_dim and cfg.song_ann:
            self.song.ann.reset()                         # (self.apply(_init) above re-initialised these with std 0.02)
        if cfg.song_dim:                                  # same for the song attention: a warm start from v3a begins exactly as v3a
            for i in self.song_layer_ids:
                nn.init.zeros_(self.dec_blocks[i].sca.proj.weight)
                nn.init.zeros_(self.dec_blocks[i].sca.proj.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ encoder
    def encode(self, mel, feats=None):
        """mel [B, 1728, 96] -> [B, 432, d]; token j is centered at tick tau_j = -768 + 4j + 2.
        feats: optional dict from the dataset / mai.feats (fsl, fi0, fw, fvalid[, keep]); a model with feature fusion and
        feats=None behaves as the mel-only fallback (the branch contributes exactly 0)."""
        x = mel.transpose(1, 2).to(self.conv1.weight.dtype)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.transpose(1, 2)
        if self.feat is not None and feats is not None:
            x = x + self.feat(feats).to(x.dtype)
        x = self.drop(x + self.enc_pos[: x.shape[1]])
        for blk in self.enc_blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(x)
        return self.enc_ln(x)

    # ------------------------------------------------------------------ song context (v3)
    def song_summary(self, song):
        """The k-independent part: run the song encoder over the whole song ONCE (inference: once per song, not per window).
        song: dict(sfeat [B,S,song_in], sscal [B,S,4], sidx [B,S][, smask [B,S] bool]).  Returns dict(mem_s [B,S,song_dim], aux [B,S,4]
        auxiliary intensity prediction (log1p notes per bar, hold / slide / touch share logits), smask [B,S] bool (column 0 forced True),
        any_valid [B] bool).  Does not depend on the window position."""
        dev = song["sfeat"].device
        smask = song.get("smask")
        smask = torch.ones(song["sfeat"].shape[:2], dtype=torch.bool, device=dev) if smask is None else smask.to(dev).bool()
        any_valid = smask.any(1)
        smask = smask.clone()
        smask[:, 0] = True                                # never an all-masked row (NaN-free softmax); such samples get keep = 0
        ann = dict(alab=song["alab"], apos=song["apos"], aend=song["aend"], aflg=song["aflg"]) if "alab" in song else None
        mem_s, aux = self.song(song["sfeat"], song["sscal"], song["sidx"], smask, ann)
        return dict(mem_s=mem_s, aux=aux, smask=smask, any_valid=any_valid)

    def song_memory(self, summ, srel, keep=None, batch=None):
        """The window-dependent part: decoder memory = projected song tokens + embedding of each bar's position relative to the window
        (srel = mai.v3.song_arrays(...)["srel"] for THIS window start bar, [B,S] or [S]).  `batch`: expand a 1-song summary to this many rows
        (the K candidate windows of one song share the song).  keep: optional [B] bool.  Returns the dict consumed by the decoder."""
        mem_s, smask, any_valid, aux = summ["mem_s"], summ["smask"], summ["any_valid"], summ["aux"]
        srel = srel.to(mem_s.device, non_blocking=bool(srel.device.type == 'cpu' and srel.is_pinned())).long()   # pinned source: async copy (speed)
        if srel.dim() == 1:
            srel = srel[None]
        B = batch if batch is not None else max(mem_s.shape[0], srel.shape[0])
        if mem_s.shape[0] != B:
            mem_s, smask, any_valid, aux = (t.expand(B, *t.shape[1:]) for t in (mem_s, smask, any_valid, aux))
        if srel.shape[0] != B:
            srel = srel.expand(B, -1)
        mem = self.song_ln(self.song_to_dec(mem_s) + self.song_rel(srel))
        keep = any_valid if keep is None else (keep.to(mem.device).bool() & any_valid)
        if self.training and self.cfg.song_p_drop > 0:
            keep = keep & (torch.rand(keep.shape[0], device=mem.device) >= self.cfg.song_p_drop)
        return dict(mem=mem, mask=smask[:, None, None, :], keep=keep.float(), aux=aux, mem_s=mem_s)

    def song_encode(self, song):
        """song: dict(sfeat, sscal, sidx, srel (+ smask [B,S] bool, optional keep [B] bool)) or an already encoded dict (has "mem").
        = song_memory(song_summary(song), srel).  Returns dict(mem [B,S,d] decoder memory, mask bool [B,1,1,S], keep [B] float, aux [B,S,4],
        mem_s [B,S,song_dim]).  keep == 0 -> the song path contributes exactly 0 for that sample (training: p = song_p_drop; always for
        samples without bar data)."""
        if song is None or "mem" in song:
            return song
        return self.song_memory(self.song_summary(song), song["srel"], song.get("keep"))

    # ------------------------------------------------------------------ decoder
    def lv_term(self, tokens):
        """smooth level embedding for LV_k tokens (zero for every other token, in particular LV_UNK): [..., d]"""
        idx = tokens - self.cfg.lv0
        m = ((idx >= 0) & (idx < self.cfg.n_lv)).unsqueeze(-1)
        if self.training or torch.is_grad_enabled():
            table = self.lv_lin(self.lv_feat)                               # [n_lv, d]
        else:
            # speed (inference only): the table depends on parameters alone -> compute once per (autocast, param version)
            key = (torch.is_autocast_enabled(), self.lv_lin.weight._version, self.lv_lin.bias._version if self.lv_lin.bias is not None else -1,
                   self.lv_feat.device)
            hit = getattr(self, "_lv_cache", None)
            if hit is None or hit[0] != key:
                hit = self._lv_cache = (key, self.lv_lin(self.lv_feat))
            table = hit[1]
        return table[idx.clamp(0, self.cfg.n_lv - 1)] * m.to(table.dtype)

    def bd_term(self, tokens, table):
        """BD ids -> table rows (zero for every other token): [..., d]"""
        idx = tokens - self.cfg.bd0
        m = ((idx >= 0) & (idx < self.cfg.n_bd)).unsqueeze(-1)
        return table(idx.clamp(0, self.cfg.n_bd - 1)) * m.to(table.weight.dtype)

    def cond_vec(self, tokens):
        """tokens [B, >=1+n_cond] -> [B, d] summary of the condition tokens (header positions 1..n_cond)"""
        c = tokens[:, 1:1 + self.cfg.n_cond]
        assert c.shape[1] == self.cfg.n_cond, "prompt must contain the whole condition header"
        e = self.cond_emb(c).sum(1)
        if self.cfg.lv_cont:
            e = e + self.lv_term(c[:, 0])
        if self.cfg.n_bd:
            e = e + self.bd_term(c, self.bd_cond).sum(1)
        return self.cond_out(self.cond_ln(e))

    def embed(self, tokens, cur_time, pos0=0, g=None):
        L = tokens.shape[1]
        if torch.is_grad_enabled() or self.training:
            pos = torch.arange(pos0, pos0 + L, device=tokens.device)
            pos_x = self.pos_emb(pos)[None]
        else:
            pos_x = self.pos_emb.weight[pos0:pos0 + L][None]              # speed: same rows as the embedding lookup of arange(pos0, pos0+L)
        phase = torch.remainder(cur_time, self.cfg.n_phase)
        beat = (torch.div(cur_time, TPB, rounding_mode="floor") + self.cfg.beat_offset).clamp(0, self.cfg.n_beat - 1)
        x = self.tok_emb(tokens) + pos_x + self.phase_emb(phase) + self.beat_emb(beat)
        if self.cfg.lv_cont:
            x = x + self.lv_term(tokens)
        if self.cfg.n_bd:
            x = x + self.bd_term(tokens, self.bd_tok)
        if g is not None:
            x = x + g[:, None]
        return x

    def deltas(self, cur_time, ref):
        dt = _autocast_dtype(ref.device.type) or ref.dtype
        d = (self.tau[None, None, :] - cur_time[:, :, None].float()) / TPB      # [B,L,S] beats, + = audio in the future
        return F.relu(d).to(dt), F.relu(-d).to(dt)

    @staticmethod
    def _causal_pad_mask(pad_mask, dtype):
        """pad_mask [B,L] bool (True = real token) -> float [B,1,L,L] causal + key padding"""
        B, L = pad_mask.shape
        causal = torch.ones(L, L, dtype=torch.bool, device=pad_mask.device).tril()
        ok = causal[None, None] & pad_mask[:, None, None, :]
        ok = ok | torch.eye(L, dtype=torch.bool, device=pad_mask.device)[None, None]   # never a fully masked row
        return torch.zeros(ok.shape, dtype=dtype, device=pad_mask.device).masked_fill(~ok, float("-inf"))

    def decode_hidden(self, tokens, cur_time, enc, pad_mask=None, song=None):
        g = self.cond_vec(tokens) if self.cfg.cond_inject else None
        x = self.drop(self.embed(tokens, cur_time, g=g))
        dpos, dneg = self.deltas(cur_time, x)
        mask = None
        if pad_mask is not None:
            mask = self._causal_pad_mask(pad_mask, _autocast_dtype(x.device.type) or x.dtype)
        for blk in self.dec_blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, enc, dpos, dneg, mask, song, use_reentrant=False)
            else:
                x = blk(x, enc, dpos, dneg, mask, song)
        return self.dec_ln(x)

    def forward(self, tokens, cur_time, mel=None, enc=None, loss_mask=None, pad_mask=None, feats=None, song=None):
        """returns logits [B,L,V]; with loss_mask [B,L] bool, only the selected positions: [N,V].
        song: raw bar-table dict or the dict returned by song_encode (pass that one to also get `aux` for the auxiliary loss); None = no song
        context (exact v3a behaviour of the decoder)."""
        if enc is None:
            enc = self.encode(mel, feats)
        song = self.song_encode(song) if (song is not None and self.cfg.song_dim) else None
        h = self.decode_hidden(tokens, cur_time, enc, pad_mask, song)
        if loss_mask is not None:
            h = h[loss_mask]
        return F.linear(h, self.tok_emb.weight)

    # ------------------------------------------------------------------ incremental decoding
    @torch.no_grad()
    def init_cache(self, enc, max_len=None, song=None):
        cfg = self.cfg
        B = enc.shape[0]
        max_len = max_len or cfg.max_len
        dt = _autocast_dtype(enc.device.type) or enc.dtype
        hd = cfg.d // cfg.heads
        ck, cv = [], []
        for blk in self.dec_blocks:
            k, v = blk.ca.make_kv(enc)
            ck.append(k.contiguous()); cv.append(v.contiguous())
        k = [enc.new_zeros(B, cfg.heads, max_len, hd, dtype=dt) for _ in self.dec_blocks]
        v = [enc.new_zeros(B, cfg.heads, max_len, hd, dtype=dt) for _ in self.dec_blocks]
        cache = DecodeCache(k, v, ck, cv)
        song = self.song_encode(song) if (song is not None and cfg.song_dim) else None
        if song is not None:                              # song-memory K/V per layer: computed once per window / song position
            cache.sk, cache.sv = [None] * len(self.dec_blocks), [None] * len(self.dec_blocks)
            for l, blk in enumerate(self.dec_blocks):
                if blk.sca is not None:
                    cache.sk[l], cache.sv[l] = [t.contiguous() for t in blk.sca.make_kv(song["mem"])]
            cache.smask, cache.skeep = song["mask"], song["keep"]
        return cache

    @torch.no_grad()
    def prefill(self, tokens, cur_time, cache, pad_mask=None):
        """run a prompt [B,L] through the decoder, fill the KV cache, return logits for the last position [B,V]"""
        assert cache.len == 0
        B, L = tokens.shape
        if self.cfg.cond_inject:
            cache.g = self.cond_vec(tokens)
        x = self.embed(tokens, cur_time, 0, g=cache.g)
        dpos, dneg = self.deltas(cur_time, x)
        mask = None
        if pad_mask is not None:
            mask = self._causal_pad_mask(pad_mask, _autocast_dtype(x.device.type) or x.dtype)
        for l, blk in enumerate(self.dec_blocks):
            x = x + blk.sa(blk.ln1(x), mask, kv_out=(cache.k[l], cache.v[l]))
            x = x + blk.ca(blk.ln2(x), cache.ck[l], cache.cv[l], dpos, dneg)
            if blk.sca is not None and cache.sk is not None:
                x = x + blk.sca(blk.ln_s(x), cache.sk[l], cache.sv[l], cache.smask, cache.skeep)
            x = x + blk.ff(blk.ln3(x))
        cache.len = L
        return F.linear(self.dec_ln(x[:, -1]), self.tok_emb.weight)

    @torch.no_grad()
    def decode_step(self, tok, cur_time, cache):
        """tok, cur_time: [B] (the token at position cache.len and its cur_time). Returns next-token logits [B,V]."""
        pos = cache.len
        assert pos < cache.k[0].shape[2]
        tok = tok.view(-1, 1)
        ct = cur_time.view(-1, 1)
        x = self.embed(tok, ct, pos, g=cache.g)
        dpos, dneg = self.deltas(ct, x)
        for l, blk in enumerate(self.dec_blocks):
            x = x + blk.sa.step(blk.ln1(x), cache.k[l], cache.v[l], pos)
            x = x + blk.ca(blk.ln2(x), cache.ck[l], cache.cv[l], dpos, dneg)
            if blk.sca is not None and cache.sk is not None:
                x = x + blk.sca(blk.ln_s(x), cache.sk[l], cache.sv[l], cache.smask, cache.skeep)
            x = x + blk.ff(blk.ln3(x))
        cache.len = pos + 1
        return F.linear(self.dec_ln(x[:, 0]), self.tok_emb.weight)

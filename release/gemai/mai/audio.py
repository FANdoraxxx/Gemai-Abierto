import subprocess, numpy as np
from .ffmpeg_util import ffmpeg_exe
SR = 24000
HOP = 120          # 200 fps
FPS = SR / HOP
NFFT = 1024
NMELS = 96

def load_audio(path, sr=SR):
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()

_mel_cache = {}
def mel_db(wave, device="cuda"):
    import torch, torchaudio
    key = str(device)
    if key not in _mel_cache:
        _mel_cache[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMELS, f_min=25.0, f_max=11000.0,
            power=2.0, center=True, norm="slaney", mel_scale="slaney").to(device)
    x = torch.from_numpy(wave).to(device)
    with torch.no_grad():
        m = _mel_cache[key](x)                      # [NMELS, T]
        m = 10.0 * torch.log10(torch.clamp(m, min=1e-10))
        m = torch.clamp(m, min=m.max() - 100.0)
    return m.T.contiguous().half().cpu().numpy()   # [T, NMELS]

def beat_sync(mel, times_sec, norm=True):
    """mel [T, M] (200 fps), times_sec [F] -> [F, M] linear interpolation; outside -> min"""
    T = mel.shape[0]
    f = times_sec * FPS
    i0 = np.floor(f).astype(np.int64)
    w = (f - i0).astype(np.float32)[:, None]
    valid = (i0 >= 0) & (i0 + 1 < T)
    i0c = np.clip(i0, 0, T - 2)
    a = mel[i0c].astype(np.float32); b = mel[i0c + 1].astype(np.float32)
    out = a * (1 - w) + b * w
    return out, valid

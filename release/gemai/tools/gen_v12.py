"""bigbase_v12: audio -> maidata.txt with a planned per-bar density (planner + writer).

    audio -> beat grid -> bar table (MERT L8 + mel per 4-beat bar, the generator's own) -> BD planner (per-bar density bucket)
          -> writer (seq_version 6, BDI tokens forced from the plan) -> out/maidata.txt (+ bd_plan.json, gen_info.json)

Defaults = the evaluated configuration (scratch/b3eval/eval_seeds_bd2.py + planner cal3_g4, K=6, quality fast, ann none).
Paths of the writer checkpoint / calibration / planner live in release/bigbase_v12/config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
ROOT = TREE.parents[1]
sys.path.insert(0, str(TREE))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
os.environ.setdefault("MAI_FEATS_ROOT", str(ROOT / "work" / "feats"))
os.environ.setdefault("HF_HOME", str(ROOT / "work" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class PlanOnlyDone(Exception):
    pass


def rel(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def parser():
    cfg = json.loads((TREE / "config.json").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--out", help="output dir (default: <repo>/out/mine/<name>_v12_<timestamp>)")
    ap.add_argument("--title"); ap.add_argument("--artist", default="")
    ap.add_argument("--level", default="auto", help="target level: 13 / 13+ / 13.7 (constant) or auto (predicted from the audio, bias -0.1); also the planner's level input")
    ap.add_argument("--slot", default="master", help="master / 5 (default) or remaster / 6")
    ap.add_argument("--seed", type=int, default=0, help="writer seed (the planner sampling is seeded with it too)")
    ap.add_argument("--device", default=cfg["device"], help="writer device (default %s)" % cfg["device"])
    ap.add_argument("--planner-device", default=cfg["planner"]["device"])
    ap.add_argument("--checkpoint", default=cfg["writer"]["checkpoint"], help="writer checkpoint (config.json: writer.checkpoint)")
    ap.add_argument("--calib-file", default=cfg["writer"]["calib_file"])
    ap.add_argument("--meta", default=cfg["writer"]["meta"])
    ap.add_argument("--k", type=int, default=cfg["writer"]["window_candidates"], help="candidates per 4-bar window (K)")
    ap.add_argument("--quality", choices=("fast", "fine", "max"), default=cfg["writer"]["quality"])
    ap.add_argument("--no-planner", action="store_true", help="BD tokens UNK (writer without a density plan)")
    ap.add_argument("--plan-file", default="", help="use this BD plan json {\"bd\":[bucket per bar]} instead of the planner")
    ap.add_argument("--plan-only", action="store_true", help="stop after the planner (writes bd_plan.json only)")
    ap.add_argument("--grid-from", help="take the tempo map of this maidata.txt / folder instead of detecting the beat grid (eval songs)")
    ap.add_argument("--bpm", type=float); ap.add_argument("--first", type=float)
    ap.add_argument("--tempo-changes", choices=("auto", "on", "off"), default="auto")
    for n in ("touch", "ex", "fest"):
        ap.add_argument(f"--{n}", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--config", default=str(TREE / "config.json"))
    return ap, cfg


def build_options(a, template):
    from mai.generate import GenerationOptions
    fields = GenerationOptions.__dataclass_fields__
    opts = GenerationOptions(**{k: v for k, v in template.items() if k in fields})
    opts.slot = "remaster" if str(a.slot).lower() in ("6", "remaster", "re:master") else "master"
    opts.grid_from = a.grid_from or None
    opts.bpm = a.bpm
    opts.first = a.first
    opts.feats = "auto"
    opts.seed = int(a.seed)
    opts.quality = a.quality
    opts.window_candidates = int(a.k)
    opts.calib_file = str(rel(a.calib_file).resolve())
    opts.level = a.level
    opts.auto_level = True
    opts.auto_level_bias = -0.1
    opts.touch, opts.ex, opts.fest = a.touch, a.ex, a.fest
    opts.version = "24"
    opts.ann = "none"
    opts.chorus = 1.0
    opts.chorus_parts = "pick,rwo"
    opts.chorus_onset_q = 0.85
    opts.tempo_changes = a.tempo_changes
    opts.brk_bias = -0.35
    opts.mod_target_scale = 1.0
    opts.brk_timing_gate = False
    opts.validate()
    return opts


def make_generator(a, cfg):
    import numpy as np
    from mai.generate import ChartGenerator, parse_level
    from mai import v3 as V3

    class V12Generator(ChartGenerator):
        """ChartGenerator whose song context gets a BD plan from the planner (computed on the SAME bar table the writer's context uses)."""
        planner = None
        plan_only = False
        last_plan = None
        plan_seed = 0

        def v3_context(self, mel, mean, std, feat_arr, tempo, first, duration, options, log=print):
            ctx, info = super().v3_context(mel, mean, std, feat_arr, tempo, first, duration, options, log=log)
            self.last_plan = None
            if ctx is None:
                log("warning: no song context (no MERT features?) -> no BD plan", flush=True)
                return ctx, info
            if self.V.seq_version < 5:
                log("warning: checkpoint is seq_version < 5: BD plan not applicable", flush=True)
                return ctx, info
            if ctx.bd is not None:
                log(f"BD plan taken from MAI_BD_OVERRIDE ({len(ctx.bd)} bars)", flush=True)
                self.last_plan = dict(bd=[int(x) for x in ctx.bd], source="MAI_BD_OVERRIDE")
            elif self.planner is not None:
                lv = parse_level(options.level)           # options.level is the effective level here (auto level already resolved)
                if lv is None:
                    lv = 13.5
                    log("warning: no level for the planner, using 13.5", flush=True)
                slot = 5 if options.slot == "master" else 6
                total_beats = float(tempo.to_beat(max(float(duration) - float(first), 0.0)))
                from mai.bdplanner import dominant_bpm
                bpm = dominant_bpm(tempo.b, tempo.p, total_beats)
                t0 = time.perf_counter()
                res = self.planner.plan(ctx.tab, bpm, lv, slot, seed=self.plan_seed)
                dt = time.perf_counter() - t0
                if res["bd"] is None:
                    log(f"warning: planner skipped: {res['reason']}", flush=True)
                else:
                    ctx.bd = np.clip(np.asarray(res["bd"], np.int64), 0, V3.N_BD_B - 1)
                    ctx.bd_override_info = dict(path="planner", n_in=int(len(ctx.bd)), n_bars=int(ctx.nb))
                    info["bd_override"] = ctx.bd_override_info
                    info["planner"] = dict(lv=lv, slot=slot, bpm=bpm, seconds=round(dt, 2), detail=res["seconds"])
                    self.last_plan = dict(bd=[int(x) for x in ctx.bd], source="planner", lv=lv, slot=slot, bpm=bpm, n_bars=int(ctx.nb),
                                          seed=self.plan_seed, seconds=round(dt, 2), edges=list(V3.BD_EDGES))
                    log(f"BD plan: {ctx.nb} bars, level {lv}, bpm {bpm:g}, mean bucket {float(np.mean(ctx.bd[ctx.bd > 0])):.2f} ({dt:.1f} s)", flush=True)
            if self.plan_only:
                raise PlanOnlyDone()
            return ctx, info

    from mai.generate import ChartGenerator as _  # noqa: F401
    eng = V12Generator(str(rel(a.checkpoint)), device=a.device, meta=str(rel(a.meta)), amp="auto")
    if not a.no_planner and not a.plan_file and cfg["planner"]["enabled"]:
        from mai.bdplanner import BDPlanner
        eng.planner = BDPlanner(rel(cfg["planner"]["dir"]), a.planner_device, threads=2)
    eng.plan_only = bool(a.plan_only)
    eng.plan_seed = int(a.seed)
    return eng


def main(argv=None):
    ap, cfg0 = parser()
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    t_all = time.perf_counter()
    audio = Path(a.audio).resolve()
    if not audio.is_file():
        sys.exit(f"audio not found: {audio}")
    title = a.title or audio.stem
    out = Path(a.out) if a.out else ROOT / "out" / "mine" / f"{audio.stem}_gemai_s8_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    if a.plan_file:
        os.environ["MAI_BD_OVERRIDE"] = str(Path(a.plan_file).resolve())
    else:
        os.environ.pop("MAI_BD_OVERRIDE", None)
    import torch
    torch.set_num_threads(1)
    template = json.loads((TREE / "planner" / "_writer_template.json").read_text(encoding="utf-8"))
    options = build_options(a, template)
    t0 = time.perf_counter()
    eng = make_generator(a, cfg)
    t_load = time.perf_counter() - t0
    print(f"[v12] audio={audio.name} level={a.level} slot={options.slot} seed={a.seed} writer={a.checkpoint} device={a.device} planner={'off' if (a.no_planner or a.plan_file) else cfg['planner']['dir']}", flush=True)
    try:
        info = eng.generate(str(audio), options, out=str(out), title=title, artist=a.artist)
    except PlanOnlyDone:
        info = None
    if eng.last_plan is not None:
        (out / "bd_plan.json").write_text(json.dumps(dict(key=title, **eng.last_plan), ensure_ascii=False), encoding="utf-8")
    total = time.perf_counter() - t_all
    summ = dict(out=str(out), model_load_s=round(t_load, 1), total_s=round(total, 1), generate_s=round(total - t_load, 1))
    if info is not None:
        for k in ("n_notes", "level_estimate", "partial", "round_trip_valid"):
            summ[k] = info.get(k)
        summ["timings"] = info.get("timings")
    print("[v12] done " + json.dumps(summ, ensure_ascii=False, default=str), flush=True)
    (out / "v12_summary.json").write_text(json.dumps(summ, ensure_ascii=False, default=str, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

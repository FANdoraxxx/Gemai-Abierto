# 模型权重许可 / Model weights license

## 模型权重：仅限非商业用途

本仓库中的以下模型权重与统计文件（下称"Gemai 权重"）以
**知识共享 署名-非商业性使用 4.0 国际许可协议（CC BY-NC 4.0）** 提供：
https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans

- `weights/gemai_s8.pt`（S8 写谱模型）
- `release/gemai/planner/` 下的疏密规划器权重（`ar_g_f*.pt`、`norm.npz`）
- `out/plan/sectionpred.pkl`（段落预测器）、`out/plan/targets.json`
- `weights/calibration_bigB_r3.json`、`release/gemai/work/` 下的各统计与校准文件

这意味着：

- **禁止商业使用**：不得将 Gemai 权重，或用它生成的谱面，用于任何以营利为目的的用途，包括但不限于出售、付费分发、付费代生成、商业游戏或商业服务。
- 可以在非商业前提下使用、复制、分享和修改，但需要注明出处（Gemai / Gemai Abierto）并附上本许可链接。
- 权重按"原样"提供，不附带任何明示或暗示的担保。

不能商业使用的另外两个原因：

1. 本项目依赖的 MERT-v1-330M 本身采用 CC BY-NC 4.0（仅限非商业）。
2. Gemai 的模型训练使用了 maimai DX 官方谱面，相关谱面的版权属于 SEGA；大底模还使用了社区自制谱。本仓库不包含任何谱面数据。

## 第三方组件

随仓库分发的第三方组件各自适用其原许可，不受本声明改变，详见 [THIRD_PARTY.md](THIRD_PARTY.md)：

- MERT-v1-330M：CC BY-NC 4.0
- FFmpeg：GPL v3
- beat_this：以上游仓库许可为准
- MaiMuriDX：上游未提供许可，公开再分发前需征得上游同意

## 代码

除第三方组件外的源代码以 MIT 许可开源，见 [LICENSE](LICENSE)。MIT 许可只适用于代码，不适用于上面列出的模型权重。

---

**English summary.** The Gemai model weights and statistics files in this repository are licensed under
CC BY-NC 4.0: **no commercial use** of the weights or of charts generated with them. Attribution is required.
Bundled third-party components keep their own licenses (MERT-v1-330M: CC BY-NC 4.0; FFmpeg: GPL v3; see THIRD_PARTY.md).
The Gemai source code is released under the MIT License (see LICENSE); the MIT License does not cover the model weights.

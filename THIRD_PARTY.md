# 第三方组件与许可

| 组件 | 用途 | 来源 | 许可 |
|---|---|---|---|
| MERT-v1-330M（随仓库分发） | 音乐特征（第 8 层） | https://huggingface.co/m-a-p/MERT-v1-330M ，固定版本 `5240c2708a5acaee1007f43fb9735c7dcd0b78c9`；作者 Yizhi Li, Ruibin Yuan, Ge Zhang 等（arXiv:2306.00107） | CC BY-NC 4.0：仅限非商业用途，需署名。权重转存为 safetensors 分片，数值逐位不变，见 `work/hf_cache/hub/models--m-a-p--MERT-v1-330M/NOTICE.md` |
| FFmpeg（随仓库分发） | 音频解码 | BtbN/FFmpeg-Builds，版本 `N-117995-g56419fd096-20241203`；源码 https://git.ffmpeg.org/ffmpeg.git | GPL v3，见 `third_party/ffmpeg/LICENSE.txt` 与 `third_party/ffmpeg/NOTICE.md` |
| beat_this（final0 检查点，随仓库分发） | 节拍跟踪 | https://github.com/CPJKU/beat_this | 以上游仓库的 LICENSE 为准 |
| MaiMuriDX 核心 | 无理检测 | Minepig/MaiMuriDX，提交 `0ebab7ff6853a159ef6af76d9e3c43b23157bb5e` | 上游未提供许可文件；详见 `release/gemai/third_party/muri-extracted/maimuridx/README.md`。**公开再分发前需向上游确认授权。** |
| PyTorch、transformers 等 Python 包 | 运行环境 | 通过 pip 安装，不随仓库分发 | 各自的许可 |

说明：
- Gemai 的写谱模型与疏密规划器训练时使用了 maimai DX 官方谱面（版权属于 SEGA）；S8 由使用过自制谱数据的大底模蒸馏而来。本仓库不包含任何谱面。
- 完整版 Gemai 另使用 MiaCode（GPLv3）做整谱无理复查；本发行版未包含它，整谱复查自动改用 MaiMuriDX。

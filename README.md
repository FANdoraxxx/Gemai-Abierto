# Gemai Abierto

从音频生成 maimai DX 谱面（`maidata.txt`）的精简发行版。写谱模型是从 Gemai 大底模Shannon蒸馏出的小模型 **S8**（约 2000 万参数）。

> ⚠️ 模型权重仅限非商业用途（CC BY-NC 4.0），见 [LICENSE.md](LICENSE.md)。

原理、训练过程和设计取舍见 [docs/Gemai_原理与技术.md](docs/Gemai_原理与技术.md)。

## 环境

- Windows 10/11（启动脚本是 .bat）
- Python 3.11
- 显卡可选：
  - 有 NVIDIA 显卡（驱动支持 CUDA 12.4）：用显卡生成，快速档一首歌约 1–2 分钟；
  - 没有 NVIDIA 显卡：自动改用 CPU，快速档一首约 3–5 分钟（实测 16 线程 CPU：MERT 特征约 2 分钟、写谱约 1.5 分钟），精细档约 3 倍。
- 约 4 GB 磁盘空间（仓库约 1.6 GB，PyTorch 另计）
- MERT 音乐模型和 ffmpeg 已经包含在仓库里，不需要另外下载。

## 安装

1. 克隆本仓库。
2. 双击 `setup.bat`。它会创建 `.venv`，并安装 PyTorch 2.6 和 `requirements.txt` 里的依赖：
   - 检测到 NVIDIA 显卡（`nvidia-smi` 可用）时装 CUDA 12.4 版；
   - 否则装 CPU 版。
   - 想强制指定：先 `set GEMAI_TORCH_INDEX=https://download.pytorch.org/whl/cpu`（或 `.../whl/cu124`）再运行。

## 使用

把音频文件（mp3 / wav / flac / ogg / m4a / mp4）拖到 `生成谱面_Gemai_S8.bat` 上，按提示选择：

- **等级**：如 `13`、`13+`、`14`、`14.5`；直接回车则按音频自动预测。
- **难度槽**：5 = MASTER，6 = Re:MASTER。
- **生成档位**：1 快速 / 2 加量 / 3 精细 / 4 极致（档位越高越慢）。
- **随机种子**：输入整数可复现结果；回车为随机。

结果在 `out\mine\<歌名>_GemaiS8_<档位>_<时间>\`，包含 `maidata.txt`、`track.mp3`、`bd_plan.json`（每小节密度规划）和 `gen_info.json`（生成记录）。

命令行用法：

```
生成谱面_Gemai_S8.bat "歌.mp3" --level 14 --slot 5 --quality fine --seed 42
```

- 设备自动选择：PyTorch 能用 NVIDIA 显卡时用 `cuda:0`，否则用 `cpu`。手动指定：先 `set MAI_DEVICE=cpu`（或 `cuda:1`）。
- CPU 模式默认使用一半的逻辑核心，可用 `set MAI_CPU_THREADS=8` 调整。
- 第一次处理一首歌时要提取 MERT 特征，会多花一些时间；结果缓存在 `release\gemai\work\feats_cache\`。

## 目录

```
生成谱面_Gemai_S8.bat     生成入口
setup.bat                 安装
release/gemai/            生成代码（mai/）、入口脚本（tools/）、疏密规划器、统计表、无理检测
scratch/plan/, out/plan/  段落预测器与统计目标
weights/                  S8 写谱模型、校准文件、beat_this 节拍模型
work/hf_cache/            MERT-v1-330M（15 个 safetensors 分片）
third_party/ffmpeg/       ffmpeg.exe 及其 DLL（解码音频）
docs/                     原理与技术文档
```

## 许可

**Gemai 的模型权重以 CC BY-NC 4.0 提供，禁止商业使用**（包括用它生成的谱面），使用时需注明出处。详见 [LICENSE.md](LICENSE.md)。

## 第三方组件

本仓库随附并使用了以下第三方组件，各自保留原许可：

- **MERT-v1-330M**（m-a-p，CC BY-NC 4.0，仅限非商业用途）：音乐特征提取。见 `work/hf_cache/hub/models--m-a-p--MERT-v1-330M/NOTICE.md`。
- **FFmpeg**（GPL v3）：音频解码。见 `third_party/ffmpeg/NOTICE.md`。
- **beat_this**（CPJKU）：节拍跟踪。
- **MaiMuriDX**：无理检测。

详细来源、许可和引用见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库不包含任何谱面数据或训练数据。

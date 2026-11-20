# Gemai Abierto

从音频生成 maimai DX 谱面（`maidata.txt`）的精简发行版。写谱模型是从 Gemai 大底模蒸馏出的小模型 **S8**（约 2000 万参数）。

原理、训练过程和设计取舍见 [docs/Gemai_原理与技术.md](docs/Gemai_原理与技术.md)。

## 环境

- Windows 10/11（启动脚本是 .bat）
- NVIDIA 显卡，驱动支持 CUDA 12.4
- Python 3.11
- ffmpeg：放在 PATH 里，或放在 `C:\ffmpeg\bin\ffmpeg.exe`，或设置环境变量 `MAI_FFMPEG` 指向 ffmpeg.exe
- 约 6 GB 磁盘空间（含 PyTorch）

## 安装

1. 克隆本仓库。
2. 双击 `setup.bat`。它会：
   - 创建 `.venv` 并安装 PyTorch 2.6（CUDA 12.4）和 `requirements.txt` 里的依赖；
   - 下载 MERT-v1-330M 权重（1.26 GB）：先尝试本仓库的 GitHub Release（需要已登录的 `gh`），失败则从 Hugging Face 下载固定版本。

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

- 默认使用 `cuda:0`；换卡：先 `set MAI_DEVICE=cuda:1`，或加 `--device cuda:1`。
- 第一次处理一首歌时要提取 MERT 特征，会多花一些时间；结果缓存在 `release\gemai\work\feats_cache\`。

## 目录

```
生成谱面_Gemai_S8.bat     生成入口
setup.bat                 安装
release/gemai/            生成代码（mai/）、入口脚本（tools/）、疏密规划器、统计表、无理检测
scratch/plan/, out/plan/  段落预测器与统计目标
weights/                  S8 写谱模型、校准文件、beat_this 节拍模型
work/hf_cache/            MERT-v1-330M（权重由 setup.bat 下载）
docs/                     原理与技术文档
```

## 第三方组件

见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库不包含任何谱面数据或训练数据。

# FFmpeg（第三方程序，随本仓库分发）

- 版本：`N-117995-g56419fd096-20241203`，Windows x64 共享库构建（BtbN/FFmpeg-Builds，gcc 14.2.0，`--enable-gpl --enable-version3 --enable-shared`）
- 许可：**GPL v3**（本构建启用了 GPL 组件），全文见同目录 `LICENSE.txt`
- 源代码：FFmpeg 源码 https://git.ffmpeg.org/ffmpeg.git （提交 `56419fd096`）；构建脚本 https://github.com/BtbN/FFmpeg-Builds
- 只包含 `ffmpeg.exe` 及其运行所需的 DLL，未修改。Gemai 只把它作为独立程序调用（解码音频），不链接其代码。

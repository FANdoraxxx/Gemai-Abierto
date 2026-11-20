# Muri 检测源码提取包

两套实现分别保留，可以单独集成，也可以用 `detect.py` 对同一谱面一起运行。输出保留各自的数据结构、默认参数和规则，不把两套诊断条数相加。

- `miacode/`：C++20 + Qt Core，原生 Simai 解析、动态模拟、静态参考和原面板结果合并逻辑。
- `maimuridx/`：Python，无渲染依赖；保留原始核心，提供 Python API 和命令行。
- `examples/`：正常谱、多押、叠键、内屏无理及带多难度的 maidata 示例。
- `detect.py`：统一调用入口，默认运行两套。

来源固定为 2026-10-29 取得的提交：

- albertoBradish/MiaCode，`hand_order_public_2_0_1`，`fc2506d41b35691c8e0bd582a82ba2c1000e8d90`。
- Minepig/MaiMuriDX，`0ebab7ff6853a159ef6af76d9e3c43b23157bb5e`。

## 直接运行

先在本目录打开 PowerShell。Python 部分已用 3.11 验证。

```powershell
# 一次运行两套，保存完整结果
python -B .\detect.py .\examples\multitouch.simai -o .\result.json

# maidata 选择 &inote_5，自动读取 first
python -B .\detect.py .\examples\maidata.txt -d 5 -o .\result.json

# 只运行 Python 引擎
python -B .\detect.py .\examples\multitouch.simai --engine maimuridx -o .\result.json

# 如需同样的手部半径，显式覆盖；两边默认值各自保留
python -B .\detect.py .\examples\multitouch.simai --hand-radius 30 -o .\result.json
```

包内 `miacode/bin/muri-cli.exe` 已在本机编译验证，slide 参考数据内嵌；需 Qt Core DLL（未打包）。统一入口也会查找 `miacode/build/Release/` 和 `miacode/build/`。其他路径通过 `--miacode-bin` 指定；Qt DLL 不在可执行文件旁或 PATH 中时，用 `--qt-bin` 指定。统一入口会识别此电脑的 `C:\Qt\6.6.3\msvc2019_64\bin`，只向子进程的 PATH 添加，不改系统设置。

```powershell
cmake -S .\miacode -B .\miacode\build -DCMAKE_PREFIX_PATH=C:/Qt/6.6.3/msvc2019_64
cmake --build .\miacode\build --config Release --parallel 4
```

具体的 C++ API、Python API、单引擎参数和资源初始化顺序，见各自目录中的 README。已有配置好的 MSVC 命令行也可选 `-G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release`。

## 输出与差异

统一结果在 `engines.miacode` 和 `engines.maimuridx` 中分别存放；运行失败会写出对应错误，退出码为 2。检测到无理本身属于成功完成，退出码为 0。两边均包含静态和动态结果；MiaCode 另外提供原有面板去重结果。字段名及单位以各自 README 为准。

两者不能期望完全一致。MiaCode 默认普通手部半径 30px，MaiMuriDX 默认 60px，均按 1080px 参考尺寸；MiaCode 默认排除 Touch 多押。相同区域松开事件的复用、重叠计数及 Touch group 的处理也有区别。`--hand-radius` 只统一半径，不会统一其他规则。

MiaCode 原入口只读取全局 `&first`；Python 封装还支持 `&first_N`，有该字段时优先使用。要让两套检测采用同一时间基准，可显式传 `--first`。输出记录各自实际使用的偏移。

原编辑器、音频、视频、网络和渲染代码不在本包中。原检测流水线内部的轨迹与判定状态结构仍保留，供检测和下游集成使用。

## 来源与许可

MiaCode 自有源码保留上游 GPLv3 及原始第三方说明，见 `miacode/LICENSE`。MaiMuriDX 固定提交没有提供 LICENSE；本包如实保留来源，不为它补设许可。两部分的许可和来源说明分别放在各自目录中。

验证的实际结果见根目录 `VERIFICATION.md`。源码提取清单记录了上游文件及哈希，便于以后核对或更新。

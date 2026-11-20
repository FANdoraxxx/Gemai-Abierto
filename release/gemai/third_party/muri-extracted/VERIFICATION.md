# 实际验证记录

验证日期：2026-10-29。环境：Windows x64、Python 3.11.9、MSVC 19.43.34808、Qt Core 6.6.3、CMake 4.0 / NMake。未改系统配置、未安装依赖、未启动服务。两个原始仓库的 `git status --porcelain` 均为空。

## MiaCode

独立静态库和 `muri-cli.exe` 的 Release 构建成功。仅链接 Qt6::Core，不需要 Qt GUI/QML/Widgets、FFmpeg、BASS 或完整编辑器。17,957,009 字节的原版 slide JSON 通过 qrc 内嵌，独立 CLI 中实际加载成功，全部 slide 用例通过。

原版 `src/tools/muri/MuriSpec.cpp` 原字节保留，通过独立资源初始化入口运行：315 条 PASS、0 条 FAIL、退出码 0。覆盖五类 Muri、头星、保护、mine、负时间、短 hold、B6 松开事件复用、连锁轨迹合并及静态/动态/面板层。CTest：`muri-spec`，1/1 通过。测试入口默认不构建，见 `miacode/README.md`。

上游清单共 65 个文件，重新逐条核对上游 SHA-256 全部匹配；交付中 64 个文件原字节一致，唯一修改的上游文件为 `src/devtools/MuriDump.cpp` 命令行封装，修改范围见 `miacode/PROVENANCE.md`。算法、parser、常量和 slide 数据未改。

已编译的 `miacode/bin/muri-cli.exe` 为 Windows x64，SHA-256：

```text
6d9e695f157ff37d414a5632538bb49602ff8f189456ab25b7ca1946a9420388
```

可执行文件嵌入 slide 数据，Qt Core 运行库未打包。在此电脑上根统一入口自动添加已存在的 Qt bin 到子进程 PATH；其他电脑需对应 Qt 和 MSVC 运行库，或自行源码构建。

本机注册的 VS2022 缺少 MSBuild，最终使用现存完整的 MSVC 编译器、Windows SDK 和 NMake 完成构建。没有修改 Visual Studio 安装。

## MaiMuriDX

主代理独立重新运行：

```powershell
python -B outputs/muri-extracted/maimuridx/verify.py --original work/MaiMuriDX
```

退出码 0，8 个 fixtures 的静态记录、动态记录和中文报告与原始 CLI 逐项一致。11 个上游文件全部字节相同。6 次重复调用、3 个并发调用、跨工作目录、maidata 难度和偏移、配置覆盖后恢复默认均通过。12 个坏输入明确拒绝；坏谱 CLI 退出 2，且不生成成功 JSON。验证前后源码和配置哈希不变。

Python 3.11.9 实际运行通过。3.10 仅做语法检查，不声称已在 3.10 运行。详细记录见 `maimuridx/VERIFICATION.md`。

## 统一入口

主代理运行 `python -B outputs/muri-extracted/verify.py`，退出码 0。7 个真实 Simai 样例全部正常运行：

- 干净谱：两边静态/动态均空。
- 多押：两边动态各 1 条 `MultiTouch`。
- 叠键：两边静态/动态各 1 条 `Overlap`。
- 内屏：两边动态各 1 条 `SlideTooFast`。
- 外键：MiaCode 静态/动态各 1 条 `SlideHeadTap`；MaiMuriDX 静态 1 条 `SlideHeadTap`、动态 1 条 `TapOnSlide`，保留原算法区别。
- 撞尾：两边静态/动态各 1 条 `TapOnSlide`。
- repeated B6：MiaCode 不报内屏，MaiMuriDX 报 1 条内屏，保留原回归差异。

另通过：maidata 的 `first=1.25` 只应用一次、显式覆盖为 2.5、两边使用 30px 半径覆盖、零 BPM/错误 slide 时值/空输入返回失败、跨目录调用、help/help-all/version、拒绝把输出写回输入且输入哈希不变。默认半径依然分别为 30px、60px。

末次只读复核发现并修复 raw Simai 注释含 `&inote_5=` 时的自动格式误判；新增双引擎和 Python API 回归，确认注释中的“三押谱”不替换实际单音符谱。

`example-both.json` 是实际运行统一入口生成的三押示例结果，含两边的完整记录。

## 可重复验证

在已安装 Qt Core、MSVC 的电脑上，可重建并运行：

```powershell
cmake -S .\miacode -B .\miacode\build -DMIACODE_MURI_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=C:/Qt/6.6.3/msvc2019_64
cmake --build .\miacode\build --config Release
$env:PATH = "C:\Qt\6.6.3\msvc2019_64\bin;$env:PATH"
ctest --test-dir .\miacode\build -C Release --output-on-failure
python -B .\verify.py
python -B .\maimuridx\verify.py
```

可选原仓库路径用于 Python 的原始 CLI 对照。原仓库不在源码交付包中；日常运行不需要它们。

验证针对提取后的实现完整性和上述样例，不证明所有谱面与真实机台判定完全一致。MaiMuriDX 原解析器仍宽容处理部分无错误日志的修饰符，封装没有重新实现完整 Simai 严格校验器；具体边界见其 README。

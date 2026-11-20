# MaiMuriDX 无界面核心提取

来源：`Minepig/MaiMuriDX`，完整提交 `0ebab7ff6853a159ef6af76d9e3c43b23157bb5e`。此目录可以单独复制使用，无需安装依赖。源码要求 Python 3.10+；上游项目声明 Python `^3.11`，本提取核心使用的 `X | Y` 类型联合、`isinstance(..., X | Y)` 和 `dataclass(slots=True)` 从 Python 3.10 可用。完整运行验证使用 Python 3.11.9；全部文件另通过 Python 3.10 语法检查。本机 3.10 启动器注册指向已不存在的解释器，未声称完成 3.10 实际运行。

`upstream/` 中七个核心模块、三个 `svg` 文件及 `config.json` 与上述提交逐字节相同；没有修改检测算法。没有携带 `main.py`、`render.py`、pygame、图片或音频。

## 立即运行

在此目录运行，或将 `cli.py`、谱面和输出路径换成绝对路径：

```powershell
python -B .\cli.py .\fixtures\overlap.simai -j .\result.json -o .\report.txt
python -B .\cli.py .\fixtures\maidata.txt -d 5 -j .\result.json -o .\report.txt
python -B .\cli.py "D:\谱面目录" -d 5 -j .\result.json -o .\report.txt
```

不传 `-j` 时，将完整 JSON 输出到标准输出。`-f` 可代替位置参数。`--first 1.25` 的单位是秒。`--hand-radius 75` 仅为这次运行覆盖普通手半径；默认始终是上游的 **60px**。`--config overrides.json` 接受只包含需要覆盖字段的 JSON 对象；未知字段会报错。

```python
# 本目录的父目录在 sys.path 中，或把完整 maimuridx 文件夹放进自己的项目。
from maimuridx import detect_simai, detect_file

result = detect_simai("(120){4}1/3/5,", first=0)
print(result["static"])    # 静态检测记录
print(result["dynamic"])   # 动态检测记录
print(result["report"])    # 原始中文报告

result = detect_file(r"D:\谱面目录\maidata.txt", difficulty=5)
comparison = detect_simai("(120){4}1-5[4:1],", hand_radius=75)
```

直接导入 `api.py` 也可使用 `from api import detect_simai`。从任意工作目录调用 API/CLI 都可以；检测内部使用绝对路径。

## 输入、结果与兼容细节

支持原始 Simai 字符串、UTF-8/BOM 文本文件以及含 `maidata.txt` 的目录。原始 Simai 直接交给上游解析器，不补逗号、不改写语法。封装先检查未结束的 Note / each 组，再单独截取解析阶段报告；发现 `Invalid`、`Unknown`、`Error` 解析诊断或异常即抛出 `DetectionError`，CLI 退出码为 2，且不写成功结果。空白/纯注释输入也拒绝。Note 必须按 Simai 语法以逗号结束所在时点；末尾可有独立 `E` 标记，例如 `(120){4}1,E`。`(120){4}1` 和 `(120){4}1/` 会明确报错，避免上游 EOF 未 flush 把 Note 丢失后误报“干净”。

这仍不是替代上游的完整 Simai 严格语法校验器；上游能够容忍且不报告错误的 Note 修饰符继续按上游处理。`metadata.parse_status` 说明已通过的检查范围。`--format simai` 或 `--format maidata` 可以关闭自动判断。

自动识别以 `maidata.txt` 文件名或去掉空行、Simai 注释后的起始 `&key=` 字段为依据。raw Simai 注释里出现 `&inote_5=` 不会触发切换格式或替换谱面。

maidata 通过 `&key=value` 分段，支持同一行多个字段；多个同名字段取最后一个。`difficulty=5` 对应 `&inote_5`，空谱不算可选难度。若仅一个非空难度则自动选择；若有多个则必须显式指定，避免误检。偏移优先级是显式 `first` / `--first` → `&first_<难度>` → 全局 `&first` → 0。没有难度独立偏移时行为与上游交互入口读取全局 `&first` 一致。报告中的行列号以所选 `inote` 内容为起点，与上游一致，并非整份 maidata 的绝对行号。

JSON 顶层 `static` 和 `dynamic` 保持原始 CLI 记录结构，附加 `report` 与 `metadata`。静态记录的 `delta` 是 **tick**，`1 tick = 1/180 秒`；动态记录的 `time` 及 `judge_areas[].time` 是**秒**。`metadata` 包含实际配置、提交、note/action 数、first 和难度。静态检测只检查叠键、外键、撞尾；动态检测还包括多押与内屏无理。静态与动态的类型可能不同，例如 `fixtures/slide_head.simai` 上游静态标记 `SlideHeadTap`，动态标记 `TapOnSlide`，此处完整保留。

## 隔离与开销

每次 API 调用启动一个独立解释器，用 `JudgeManager.tick(1)` 模拟，和上游 CLI 一样。上游的全局 `REPORT_WRITER`、Slide/Wifi 注册表与配置常量不能跨配置安全共享；不建议在宿主解释器直接导入 `upstream` 下的模块或复用它们的全局状态。

封装 API 可以并发调用，每次调用都有独立子进程；验证包含三个线程同时调用。`core.py` 原本会在配置缺失时自动向源码旁写 `config.json`；本封装将不变的 `core.py` 与本次配置复制到临时目录，再从临时目录加载。其他模块只读，禁用字节码写入；运行后删除临时目录。调用者的当前目录、提取源码和配置均不被修改。CLI 仅在指定 `-j` / `-o` 时写用户要求的输出，并拒绝覆盖输入、配置和源码。

默认超时 120 秒，可用 API 的 `timeout=None` 取消，CLI 用 `--timeout` 调整；时间取决于整张谱的时间跨度、note 数和 `first`。这不是实时渲染器，不需要服务或重启。

## 源文件位置

- `upstream/core.py`：判定常量、区域几何、配置、报告缓冲。
- `upstream/action.py`：按压/划动/外键动作模拟。
- `upstream/simai.py`：Note 状态与判定逻辑。
- `upstream/slides.py`：轨迹几何和判定区信息。
- `upstream/majparse.py`：Simai/MA2 解析器与 Note→Action 转换；本封装只暴露 Simai/maidata。
- `upstream/judge.py`：静态检查、触点合并、多押与动态判定。
- `upstream/util.py`：TouchGroup 最小覆盖圆。
- `upstream/svg/`：随原库内嵌的 SVG 路径解析与几何计算。
- `upstream/config.json`：原提交实际使用的默认配置。
- `api.py` / `_worker.py` / `cli.py`：输入选择、进程隔离及非交互入口。
- `verify.py` / `fixtures/`：可重复运行的正反例与上游一致性检查。

## 原始配置单位

几何长度以 **1080×1080 原始框体坐标的像素**填写；核心内部按 540×540 画布缩放 0.5 倍。角度单位是度。以下配置均在原始配置中保留：

- 像素：`hand_radius_max=180`、`hand_radius_wifi=100`、`hand_radius_normal=60`、`distance_merge_slide=20`。
- 度：`delta_tangent_merge_slide=3`。
- **60 FPS 帧**：`tap_on_slide_threshold=0.3333`、`touch_on_slide_threshold=8`、`overlay_threshold=2`、`collide_threshold=12`、`extra_paddown_delay=3`、`release_delay=1`。
- 布尔：`wifi_need_c=false`。

`JUDGE_TPF=3`、`JUDGE_TPS=180`；以上帧配置加载时乘以 3 转为内部 tick。不要把帧值直接当 tick。特别是 `release_delay` 应以实际源配置的 **1 帧**为准，旧版 README/开发笔记提到的 1.3333 帧没有用于此提交。

## 复核

```powershell
python -B .\verify.py
python -B .\verify.py --original "D:\源码\MaiMuriDX"
```

第二条另外核对完整提交、全部上游文件字节与原始 `cli.py` 的逐例 static/dynamic JSON 和报告文本。测试包含干净谱、叠键、多押、内无、外键、撞尾、Touch、Wifi、重复调用、线程并发、跨目录、maidata 难度/偏移、一次性手半径覆盖和源码/配置不变。另有 12 个坏输入拒绝测试、常见末尾 `E` 兼容测试，以及坏谱 CLI 退出 2 且不生成 JSON 的检查。

## 授权情况

该提交中没有找到 `LICENSE` / `COPYING` 文件，也没有在所提取 Python 源文件（包括内嵌 `svg`）中找到独立许可或版权声明。这里保留原始源文件全部内容与出处，不将其声称为 MIT、BSD 或其他已确认的开放许可。此为源代码提取与技术验证，公开再分发或整合前应向上游确认授权；此说明不授予上游代码的新许可。

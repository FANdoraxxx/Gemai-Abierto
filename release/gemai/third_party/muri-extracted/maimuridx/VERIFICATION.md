# 验证记录

TOKEN=MMDX-20261029-7136  
TOKEN=MMPARSE-20261029-5419

源提交：`0ebab7ff6853a159ef6af76d9e3c43b23157bb5e`。运行时：Windows / Python 3.11.9。

已执行：

```powershell
python -B outputs/muri-extracted/maimuridx/verify.py --original work/MaiMuriDX
python -B outputs/muri-extracted/maimuridx/cli.py outputs/muri-extracted/maimuridx/fixtures/overlap.simai -j outputs/muri-extracted/maimuridx/example-result.json -o outputs/muri-extracted/maimuridx/example-report.txt
```

第一条返回码 0；自动检查结果：

- `core_byte_identity=true`：11 个文件全部与源提交逐字节相等，哈希见 `UPSTREAM-SHA256.json`。
- `original_cli_parity=8`：以下八例的 static、dynamic JSON 记录及中文报告文本与原始 CLI 完全相同。
- `repeated_calls=6`、`parallel_calls=3`：报告不积累，独立配置不串值。
- `cwd_independence=true`：在新临时空目录按绝对路径导入 API、运行 CLI；API 未向当前目录写任何文件。
- `config_and_source_unchanged=true`：验证前后对全部交付源码、配置、fixtures 的哈希快照相同。
- `maidata_difficulty_and_first=true`：多难度必须选；选择 4 为干净谱，5 为多押谱；全局 first=1.25、难度独立 first=2.5 和显式覆盖均验证动态时间。
- `invalid_inputs_rejected=12`：空/空白/纯注释、未结束 Note、未结束 each、非法 slide 时值、零 BPM、非法符号、未知 slide 形状、Note 后缺分隔即控制命令。
- `conventional_final_E_accepted=true`；`parse_failure_cli_exit2_no_result=true`。
- `maidata_fields_in_simai_comments_ignored=true`：Simai 注释中的 `&inote_5=` 不切换为 maidata，不替换实际谱面。
- `python_3_10_grammar=true`：全部 Python 文件通过 3.10 语法检查。实际 3.10 解释器注册已失效，未声称实际运行 3.10。

八个 fixtures 的原始类别：

- `clean`：S 空，D 空，4 notes。
- `multitouch`：S 空，D MultiTouch，3 notes。
- `overlap`：S Overlap，D Overlap，2 notes。
- `slide_fast`：S 空，D SlideTooFast，4 notes。
- `slide_head`：S SlideHeadTap，D TapOnSlide，3 notes。
- `tap_on_slide`：S TapOnSlide，D TapOnSlide，3 notes。
- `touch`：S 空，D MultiTouch，3 notes。
- `wifi`：S 空，D 空，2 notes。

S 为静态，D 为动态。保留上下游类别差异；未改变原始算法或 60px 默认普通手半径。只在封装层拒绝上游明确日志失败或会在 EOF 丢失的输入，不自动修谱。

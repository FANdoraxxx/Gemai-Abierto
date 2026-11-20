@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set CUDA_DEVICE_ORDER=PCI_BUS_ID
if exist "C:\ffmpeg\bin\ffmpeg.exe" set "PATH=C:\ffmpeg\bin;%PATH%"
set "MAI_FEATS_ROOT=%~dp0work\feats"
set "HF_HOME=%~dp0work\hf_cache"
set "TORCH_HOME=%~dp0weights\torch"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
REM Gemai Abierto S8 = BD density planner + 20M-parameter distilled writer.
REM GPU: cuda:0 by default. To change: set MAI_DEVICE=cuda:1 before running, or pass --device cuda:1 on the command line.
REM Python: MAI_PYTHON if set, else .venv\Scripts\python.exe (created by setup.bat), else python on PATH.
if not defined MAI_DEVICE set "MAI_DEVICE=cuda:0"
set "R=%~dp0release\gemai"
if not defined MAI_PYTHON (
  if exist "%~dp0.venv\Scripts\python.exe" (set "MAI_PYTHON=%~dp0.venv\Scripts\python.exe") else (set "MAI_PYTHON=python")
)
if not exist "%~dp0work\hf_cache\hub\models--m-a-p--MERT-v1-330M\snapshots\5240c2708a5acaee1007f43fb9735c7dcd0b78c9\pytorch_model.bin" (
  echo 缺少 MERT 权重。请先运行 setup.bat。
  pause
  exit /b 1
)

if "%~1"=="" (
  echo 用法：把音频文件（mp3/wav/flac/ogg/m4a/mp4）拖到本文件上；或在下面直接输入/拖入音频路径。
  echo 命令行：  生成谱面_Gemai_S8.bat "歌.mp3" --level 14 --slot 5     可加 --quality fast/fine/max --k 6 --seed 42
  echo.
  set "AUDIO="
  set /p "AUDIO=音频文件路径： "
  if not defined AUDIO exit /b
  set "AUDIO=!AUDIO:"=!"
  set "ASK=1"
) else (
  set "AUDIO=%~1"
  set "ASK="
  if "%~2"=="" set "ASK=1"
)
if not exist "!AUDIO!" (
  echo 找不到音频文件：!AUDIO!
  pause
  exit /b
)
for %%f in ("!AUDIO!") do set "NAME=%%~nf"

set "LEVEL=auto"
set "SLOT=5"
set "TIER=1"
set "SEED=-1"
if defined ASK (
  echo.
  set /p "LEVEL=目标等级（如 13 / 13+ / 14 / 14+ / 14.5；直接回车 = 自动按音频预测）： "
  if not defined LEVEL set "LEVEL=auto"
  set /p "SLOT=难度（5 = MASTER，6 = Re:MASTER；直接回车 = 5）： "
  if not defined SLOT set "SLOT=5"
  type "%~dp0launcher\tier_menu.txt"
  set /p "TIER=请选择 1/2/3/4（直接回车 = 1）： "
  if not defined TIER set "TIER=1"
  set /p "SEED=随机种子（输入整数可固定生成结果；直接回车 = -1 随机）： "
  if not defined SEED set "SEED=-1"
  set "EXTRA="
) else (
  set "EXTRA=%2 %3 %4 %5 %6 %7 %8 %9"
)

set "QARGS=--quality fast --k 6"
set "TIERNAME=快速"
if "!TIER!"=="2" set "QARGS=--quality fast --k 12"
if "!TIER!"=="2" set "TIERNAME=加量"
if "!TIER!"=="3" set "QARGS=--quality fine"
if "!TIER!"=="3" set "TIERNAME=精细"
if "!TIER!"=="4" set "QARGS=--quality max"
if "!TIER!"=="4" set "TIERNAME=极致"
if not defined ASK set "TIERNAME=命令行"

for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%t"
set "OUT=%~dp0out\mine\!NAME!_GemaiS8_!TIERNAME!_%TS%"
echo.
if defined ASK (
  echo 正在生成：!NAME!    等级=!LEVEL!  难度槽=!SLOT!  档位=!TIERNAME!  种子=!SEED!  显卡=%MAI_DEVICE%
  echo 输出目录：!OUT!
  "!MAI_PYTHON!" "%R%\tools\gen_v12.py" "!AUDIO!" --out "!OUT!" --title "!NAME!" --level "!LEVEL!" --slot "!SLOT!" --seed !SEED! --device %MAI_DEVICE% !QARGS!
) else (
  echo 正在生成：!NAME!    参数=!EXTRA!  显卡=%MAI_DEVICE%
  echo 输出目录：!OUT!
  "!MAI_PYTHON!" "%R%\tools\gen_v12.py" "!AUDIO!" --out "!OUT!" --title "!NAME!" --device %MAI_DEVICE% !EXTRA!
)
if exist "!OUT!\maidata.txt" (
  echo.
  echo 完成！
  if not defined MAI_NO_EXPLORER explorer "!OUT!"
) else (
  echo.
  echo 生成失败，请检查上方报错。
)
if not defined MAI_NO_PAUSE pause

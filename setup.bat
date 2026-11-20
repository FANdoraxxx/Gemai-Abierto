@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
REM Gemai Abierto setup: Python venv + packages. MERT and ffmpeg are already in the repository.
REM Needs Python 3.11 (py launcher or python on PATH). NVIDIA GPU optional: without one, the CPU build of PyTorch is installed.

if exist ".venv\Scripts\python.exe" goto venv_ok
echo [1/2] 创建虚拟环境 .venv
py -3.11 -m venv .venv 2>nul || python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo 创建虚拟环境失败：请先安装 Python 3.11。
  pause
  exit /b 1
)
:venv_ok
".venv\Scripts\python.exe" -m pip install --upgrade pip
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_KIND=CPU"
where nvidia-smi >nul 2>nul && nvidia-smi >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu124" && set "TORCH_KIND=CUDA 12.4"
if defined GEMAI_TORCH_INDEX set "TORCH_INDEX=%GEMAI_TORCH_INDEX%"
if defined GEMAI_TORCH_INDEX set "TORCH_KIND=%GEMAI_TORCH_INDEX%"
echo [2/2] 安装 PyTorch（%TORCH_KIND%）和 requirements.txt
".venv\Scripts\python.exe" -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url %TORCH_INDEX% || goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
echo.
echo 安装完成（PyTorch：%TORCH_KIND%）。把音频拖到 生成谱面_Gemai_S8.bat 上即可生成谱面。
pause
exit /b 0
:fail
echo.
echo 安装失败，请检查上方报错。
pause
exit /b 1

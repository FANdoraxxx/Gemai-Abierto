@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
REM Gemai Abierto setup: Python venv + packages. MERT and ffmpeg are already in the repository.
REM Requires 64-bit CPython 3.11 or 3.12. NVIDIA GPU optional.

if exist ".venv" goto check_venv
echo [1/2] 创建虚拟环境 .venv
if defined MAI_PYTHON (
  call :create_venv "%MAI_PYTHON%" || goto bad_python
  goto check_venv
)
call :create_venv py -3.11 && goto check_venv
call :create_venv py -3.12 && goto check_venv
call :create_venv python || goto bad_python

:check_venv
".venv\Scripts\python.exe" -c "import sys, sysconfig; sys.exit(0 if sys.implementation.name == 'cpython' and sys.version_info[:2] in ((3, 11), (3, 12)) and sysconfig.get_platform() == 'win-amd64' else 1)" 2>nul || goto bad_venv
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_KIND=CPU"
where nvidia-smi >nul 2>nul && nvidia-smi >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu124" && set "TORCH_KIND=CUDA 12.4"
if "%TORCH_KIND%"=="CUDA 12.4" (
  for /f "tokens=1 delims=. " %%g in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
    if "%%g"=="12" (
      set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
      set "TORCH_KIND=CUDA 12.8"
    )
  )
)
if defined GEMAI_TORCH_INDEX set "TORCH_INDEX=%GEMAI_TORCH_INDEX%"
if defined GEMAI_TORCH_INDEX set "TORCH_KIND=%GEMAI_TORCH_INDEX%"
set "TORCH_PACKAGES=torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0"
if /i "%TORCH_INDEX%"=="https://download.pytorch.org/whl/cu128" set "TORCH_PACKAGES=torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0"
echo [2/2] 安装 PyTorch（%TORCH_KIND%）和 requirements.txt
".venv\Scripts\python.exe" -m pip install %TORCH_PACKAGES% --index-url "%TORCH_INDEX%" || goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
echo.
echo 安装完成（PyTorch：%TORCH_KIND%）。把音频拖到 生成谱面_Gemai_S8.bat 上即可生成谱面。
pause
exit /b 0

:create_venv
%* -c "import sys, sysconfig; sys.exit(0 if sys.implementation.name == 'cpython' and sys.version_info[:2] in ((3, 11), (3, 12)) and sysconfig.get_platform() == 'win-amd64' else 1)" 2>nul || exit /b 1
%* -m venv .venv
exit /b %errorlevel%

:bad_python
echo 请先安装 64 位 CPython 3.11 或 3.12，也可以设置 MAI_PYTHON 指向已有的 python.exe。
pause
exit /b 1

:bad_venv
echo 现有 .venv 无法启动或 Python 版本不受支持。请将 .venv 重命名后重新运行 setup.bat。
pause
exit /b 1

:fail
echo.
echo 安装失败，请检查上方报错。
pause
exit /b 1
